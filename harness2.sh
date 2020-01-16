rep() {
	ls -d ./experiment/$1/* | while read i
	do
		echo $i
		(cd $i && time (python /nethome/sheneman/src/lip/train.py --config=train.yaml > train_output.txt) 2>> train_output.txt)
		(cd $i && time (python /nethome/sheneman/src/lip/classify.py --config=classify.yaml > classify_output.txt) 2>> classify_output.txt)
		(cd $i && time (python /nethome/sheneman/src/lip/score.py --config=score.yaml > scores.csv))
	done
}


ls ./experiment/ | while read i
do
	rep $i &
done

