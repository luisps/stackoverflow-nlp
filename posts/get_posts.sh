#!/bin/bash

#Selecting region
if [ -z $1 ]; then
	region="pt"
	echo "No region passed as argument. Using default region: $region"
else
	if [[ ! $1 =~ ^(en|pt|es|ru|ja)$ ]]; then 
		echo "Region must be either en, pt, es, ru or ja."
		exit
	fi
	region=$1
fi

out_file="Posts_$region.xml"
if [ -f $out_file ]; then
	echo "$out_file already exists"
	exit
fi

#compressed file name
if [ $region == 'en' ]; then
	zipped_file="stackoverflow.com-Posts.7z"
	echo "Large file warning: Compressed Posts.xml for en is ~10GB. Uncompressed is ~55GB."
else
	zipped_file="$region.stackoverflow.com.7z"
fi

#download compressed file if it doesn't already exist locally
if [ ! -f $zipped_file ]; then
	download_link="https://archive.org/download/stackexchange/$zipped_file"
	wget $download_link -q --show-progress
fi

echo "Uncompressing $zipped_file"
7za -y x $zipped_file Posts.xml > /dev/null

mv Posts.xml $out_file
rm $zipped_file

echo "$out_file is ready"
