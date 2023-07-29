#!/bin/bash



read -r -p "输入用户名，邮箱，git地址，准备好了没，[Y/n] " input

git config --global user.name $1
git config --global user.email $2

git init
git add .
git commit -m .
git remote add origin $3
git push -u origin master

read -r -p "Exit ？？[Y/n] " input