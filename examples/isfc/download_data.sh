#!/bin/sh
curl --location -o isfc.zip https://api.onedrive.com/v1.0/shares/s!Aobi2ryypFQCgqQOm2Zhs-TSLu9leQ/root/content
unzip -qo isfc.zip
rm -f isfc.zip
