#!/bin/bash

read -r -p "准备好了没，[Y/n] " input

git branch $1
git branch 
git checkout $1
git branch
git status
git checkout master
git merge $1
git push -u origin master

read -r -p "Exit ？？[Y/n] " input