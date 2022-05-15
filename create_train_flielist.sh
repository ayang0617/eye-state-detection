# shellcheck disable=SC1113
# /usr/bin/env sh
DATA=/Users/ayang/PycharmProjects/pythonProject/EyesImageDatabase
DATA1=/Users/ayang/PycharmProjects/pythonProject
echo "Create filelist.txt"
rm -rf $DATA/filelist.txt
echo "finding...."
find $DATA -name 'closed_eyes01*.jpg' | cut -d '/' -f1-15 | sed "s/$/ 0/">>$DATA1/filelist.txt
find $DATA -name 'opened_eyes01*.jpg' | cut -d '/' -f1-15 | sed "s/$/ 1/">>$DATA1/tmp0.txt

echo "find over..."
cat $DATA1/tmp0.txt>>$DATA1/filelist.txt

rm -rf $DATA1/tmp0.txt


echo "Done.."
