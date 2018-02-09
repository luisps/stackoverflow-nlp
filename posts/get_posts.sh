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

mkdir -p xml-data
posts_file="xml-data/Posts_$region.xml"
users_file="xml-data/Users_$region.xml"
if [ -f $posts_file ] && [ -f $users_file ]; then
	echo "$posts_file and $users_file already exist"
	exit
fi

#download compressed files
download_link="https://archive.org/download/stackexchange/"

if [ $region == 'en' ]; then
	zipped_posts="stackoverflow.com-Posts.7z"
	zipped_users="stackoverflow.com-Users.7z"
	echo "Large file warning: Compressed Posts.xml for EN is ~15GB. Uncompressed is ~60GB"

	wget $download_link$zipped_posts -q --show-progress
	wget $download_link$zipped_users -q --show-progress

	echo "Uncompressing $posts_file"
	7za -y x $zipped_posts Posts.xml > /dev/null

	echo "Uncompressing $users_file"
	7za -y x $zipped_users Users.xml > /dev/null

	mv Posts.xml $posts_file
	mv Users.xml $users_file

	echo "Removing file $zipped_posts"
	rm $zipped_posts

	echo "Removing file $zipped_users"
	rm $zipped_users

else
	zipped_file="$region.stackoverflow.com.7z"
	wget $download_link$zipped_file -q --show-progress

	echo "Uncompressing $posts_file"
	7za -y x $zipped_file Posts.xml > /dev/null

	echo "Uncompressing $users_file"
	7za -y x $zipped_file Users.xml > /dev/null

	mv Posts.xml $posts_file
	mv Users.xml $users_file

	echo "Removing file $zipped_file"
	rm $zipped_file

fi

