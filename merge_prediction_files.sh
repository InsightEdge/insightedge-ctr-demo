echo "Merging files in $1"

rm -rf $1_merged.txt
echo "id,click" >> $1_merged.txt

for filename in $1/*; do
    cat $filename >> $1_merged.txt
done