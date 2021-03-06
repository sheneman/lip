#!/usr/bin/bash

ls -d experiment/* | while read x
do
    python ./split.py
    head -1 validationlist.txt > v.txt
    echo "python ./split.py"
    shuf -n 1000 testlist.txt > foo.txt
    echo "shuf -n 1000 testlist.txt > foo.txt"

    ls -d $x/* | while read y
    do
	cp foo.txt $y/testlist.txt
	echo "cp foo.txt $y/testlist.txt"
	
	cp v.txt $y/validationlist.txt
	echo "cp v.txt $y/validationlist.txt"
    done

    for i in 100 250 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
    do
	shuf -n $i trainlist.txt > $x/$i/trainlist.txt
	echo "shuf -n $i trainlist.txt > $x/$i/trainlist.txt" 
    done

done

