#!/bin/bash
file=$1

if [ -e ${file} ]
then
	rsync -avzhe ssh ${file} bicsu@ssh.pythonanywhere.com:PoCheck/attend &
else
	echo "No such file!!!!!"
fi
