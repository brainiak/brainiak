"""
A pytest plugin to run tests with mpiexec who are marked with the `mpiexec`
marker.

Taken from: https://github.com/minrk/pytest-mpiexec

Slightly modified to work with BrainIAK's pytest configuration.

"""

import json
import os
import shlex
import subprocess
import sys
from enum import Enum
from functools import partial
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pytest_reportlog.plugin import ReportLogPlugin

MPI_SUBPROCESS_ENV = "TEST_MPI_SUBTEST"
TEST_REPORT_DIR_ENV = "TEST_MPI_REPORT_DIR"

MPI_MARKER_NAME = "mpiexec"

MPIEXEC = "mpiexec"


def pytest_addoption(parser):
    group = parser.getgroup("mpiexec")
    group.addoption(
        "--mpiexec",
        action="store",
        dest="mpiexec",
        default=MPIEXEC,
        help="Executable for running MPI, default=mpiexec",
    )
    group.addoption(
        "--mpiexec-report",
        action="store",
        dest="mpiexec_report",
        choices=[r.value for r in ReportStyle],
        default=ReportStyle.first_failure,
        help="""style of mpi error reporting.

        Since each mpi test represents one test run per rank,
        there are lots of ways to represent a failed parallel run:

        Options:

        - first_failure (default): report only one result per test,
          PASSED or FAILED, where FAILED will be the failure of the first
          rank that failed.
        - all_failures: report failures from all ranks that failed
        - all: report all results, including all passes
        - concise: like first_failure, but try to report all _unique_
          failures (experimental)
        """,
    )


def pytest_configure(config):
    global MPIEXEC
    global REPORT_STYLE
    mpiexec = config.getoption("mpiexec")
    if mpiexec:
        MPIEXEC = mpiexec

    REPORT_STYLE = config.getoption("mpiexec_report")

    config.addinivalue_line("markers",
                            f"{MPI_MARKER_NAME}: Run this text with mpiexec")
    if os.getenv(MPI_SUBPROCESS_ENV):
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.rank
        reportlog_dir = Path(os.getenv(TEST_REPORT_DIR_ENV, ""))
        report_path = reportlog_dir / f"reportlog-{rank}.jsonl"
        config._mpiexec_reporter = reporter = (
            ReportLogPlugin(config, report_path))
        config.pluginmanager.register(reporter)


def pytest_unconfigure(config):
    reporter = getattr(config, "_mpiexec_reporter", None)
    if reporter:
        reporter.close()


def mpi_runtest_protocol(item):
    """The runtest protocol for mpi tests

    Runs the test in an mpiexec subprocess

    instead of the current process
    """
    hook = item.config.hook
    hook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    call = pytest.CallInfo.from_call(
        partial(mpi_runtest, item), "setup")
    if call.excinfo:
        report = hook.pytest_runtest_makereport(item=item, call=call)
        hook.pytest_runtest_logreport(report=report)
    hook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)


def pytest_runtest_protocol(item, nextitem):
    """Run the MPI protocol for mpi tests

    otherwise, do nothing
    """
    if os.getenv(MPI_SUBPROCESS_ENV):
        return
    mpi_mark = item.get_closest_marker(MPI_MARKER_NAME)
    if not mpi_mark:
        return
    mpi_runtest_protocol(item)
    return True


class ReportStyle(Enum):
    all = "all"
    all_failures = "all_failures"
    first_failure = "first_failure"
    concise = "concise"


def _report_key(report):
    """Determine if a given report has been 'seen' before"""
    # use reprcrash for 'same' error message

    message_key = None
    if report["outcome"] != "passed":
        # for failures, use first line of reprcrash
        # (i.e. the line used)
        longrepr = report["longrepr"]
        if longrepr:
            reprcrash = longrepr["reprcrash"]
            if reprcrash:
                message_key_items = []
                for key, value in sorted(
                        report["longrepr"]["reprcrash"].items()):
                    if key == "message":
                        value = value.splitlines()[0]
                    message_key_items.append((key, value))
                message_key = tuple(message_key_items)

    if not message_key and report["outcome"] != "passed":
        # warn about missing message key?
        # warnings.warn("Expected reprcrash...", RuntimeWarning, stacklevel=2)
        pass
    return (report["when"], report["outcome"], message_key)


def consolidate_reports(nodeid, reports, style=ReportStyle.first_failure):
    """Consolidate a collection of TestReports

    - collapses to single success if all succeed
    - all_failures reports all failures
    - first_failure reports only the first failure
    """
    style = ReportStyle(style)

    all_ranks = {report["_mpi_rank"] for report in reports}
    if len(all_ranks) == 1:
        # only one rank, nothing to consolidate
        return reports

    if (style != ReportStyle.all and
            all(r["outcome"] == "passed" for r in reports)):
        # report from rank 0 if everything passed
        return [report for report in reports if report["_mpi_rank"] == 0]

    failed_ranks = set()
    for report in reports:
        rank = report["_mpi_rank"]
        # add rank to labels for ranks after 0, unless reporting all failures
        if rank > 0 or style in {ReportStyle.all, ReportStyle.all_failures}:
            report["nodeid"] = f"{nodeid} [rank={rank}]"
            report["location"][-1] = report["location"][-1] + f" [rank={rank}]"
        if report["outcome"] != "passed":
            failed_ranks.add(report["_mpi_rank"])
    failed_ranks = sorted(failed_ranks)

    if style == ReportStyle.all:
        return reports

    elif style == ReportStyle.all_failures:
        # select all reports on failed ranks
        return [r for r in reports if r["_mpi_rank"] in failed_ranks]

    elif style == ReportStyle.first_failure:
        # return just the first error
        first_failed_rank = failed_ranks[0]

        return [r for r in reports if r["_mpi_rank"] == first_failed_rank]
    elif style == ReportStyle.concise:
        # group by 'unique' reports
        reports_by_rank = {}
        for report in reports:
            reports_by_rank.setdefault(report["_mpi_rank"], []).append(report)
        _seen_keys = {}
        collected_reports = []
        for rank, rank_reports in reports_by_rank.items():
            rank_key = tuple(_report_key(report) for report in rank_reports)
            if rank_key in _seen_keys:
                _seen_keys[rank_key].append(rank)
            else:
                _seen_keys[rank_key] = [rank]
                collected_reports.extend(rank_reports)
        return collected_reports
    else:
        raise ValueError(f"Unhandled ReportStyle: {style}")

    return reports


def mpi_runtest(item):
    """Replacement for runtest

    Runs a single test with mpiexec
    """
    mpi_mark = item.get_closest_marker(MPI_MARKER_NAME)
    # allow parametrization
    if getattr(item, "callspec", None) and "mpiexec_n" in item.callspec.params:
        n = item.callspec.params["mpiexec_n"]
    else:
        n = mpi_mark.kwargs.get("n", 2)
    timeout = mpi_mark.kwargs.get("timeout", 120)
    exe = [
        MPIEXEC,
        "-n",
        str(n),
        sys.executable,
        "-m",
        "pytest",
        "--quiet",
        "--no-header",
        "--no-summary",
        f"{item.fspath}::{item.name}",
    ]
    env = dict(os.environ)
    env[MPI_SUBPROCESS_ENV] = "1"
    # add the mpiexec command for easy re-run
    item.add_report_section(
        "setup", "mpiexec command", f"{MPI_SUBPROCESS_ENV}=1 {shlex.join(exe)}"
    )

    with TemporaryDirectory() as reportlog_dir:
        env[TEST_REPORT_DIR_ENV] = reportlog_dir
        try:
            p = subprocess.run(
                exe,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            if e.stdout:
                item.add_report_section(
                    "mpiexec pytest", "stdout",
                    e.stdout.decode("utf8", "replace")
                )
            if e.stderr:
                item.add_report_section(
                    "mpiexec pytest", "stderr",
                    e.stderr.decode("utf8", "replace")
                )
            pytest.fail(
                f"mpi test did not complete in {timeout} seconds",
                pytrace=False,
            )

        # Collect logs from all ranks
        reports = {}
        for rank in range(n):
            reportlog_file = os.path.join(reportlog_dir,
                                          f"reportlog-{rank}.jsonl")
            if os.path.exists(reportlog_file):
                with open(reportlog_file) as f:
                    for line in f:
                        report = json.loads(line)
                        if report["$report_type"] != "TestReport":
                            continue
                        report["_mpi_rank"] = rank
                        nodeid = report["nodeid"]
                        reports.setdefault(nodeid, []).append(report)

        for nodeid, report_list in reports.items():
            # consolidate reports according to config
            reports[nodeid] = consolidate_reports(
                nodeid, report_list, REPORT_STYLE)

        # collect report items for the test
        for report in chain(*reports.values()):
            if report["$report_type"] == "TestReport":
                # reconstruct and redisplay the report
                r = item.config.hook.pytest_report_from_serializable(
                    config=item.config, data=report
                )
                item.config.hook.pytest_runtest_logreport(
                    config=item.config, report=r)

    if p.returncode or not reports:
        if p.stdout:
            item.add_report_section("mpiexec pytest", "stdout", p.stdout)
        if p.stderr:
            item.add_report_section("mpiexec pytest", "stderr", p.stderr)
    if not reports:
        pytest.fail("No test reports captured from mpi subprocess!",
                    pytrace=False)
