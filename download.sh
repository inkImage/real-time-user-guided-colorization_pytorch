FILE=$1

if [ $FILE == 'CelebA_FD' ]
then
    URL=https://www.dropbox.com/s/e0ig4nf1v94hyj8/CelebA.zip?dl=0
    ZIP_FILE=./data/CelebA_FD.zip
elif [ $FILE == 'CelebA' ]
then
    URL=https://www.dropbox.com/s/3e5cmqgplchz85o/CelebA_nocrop.zip?dl=0
    ZIP_FILE=./data/CelebA.zip
elif [ $FILE == 'LSUN' ]
then
    URL=https://www.dropbox.com/s/zt7d2hchrw7cp9p/church_outdoor_train_lmdb.zip?dl=0
    ZIP_FILE=./data/church_outdoor_train_lmdb.zip
else
    echo "Available datasets are: CelebA, CelebA_FD and LSUN"
    exit 1
fi

mkdir -p ./data/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/

if [ $FILE == 'CelebA' ]
then
    mv ./data/CelebA_nocrop ./data/CelebA
elif [ $FILE == 'CelebA_FD' ]
then
    mv ./data/CelebA ./data/CelebA_FD
fi

rm $ZIP_FILE
