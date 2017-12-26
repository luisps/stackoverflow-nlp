#!/bin/bash

#making sure p7zip is installed - needed for uncompression
if ! hash 7za 2>/dev/null; then
	echo "Unable to unzip. Please install p7zip-full package"
	exit
fi

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
	zipped_file="stackoverflow.com-PostHistory.7z"
	echo "Large file warning: Compressed PostHistory.xml for EN is ~20GB."
else
	zipped_file="$region.stackoverflow.com.7z"
fi

#download compressed file if it doesn't already exist locally
if [ ! -f $zipped_file ]; then
	download_link="https://archive.org/download/stackexchange/$zipped_file"
	wget $download_link -q --show-progress
fi

echo "Uncompressing $zipped_file"
7za -y x $zipped_file PostHistory.xml > /dev/null

mv PostHistory.xml $out_file
rm $zipped_file

echo "$out_file is ready"
