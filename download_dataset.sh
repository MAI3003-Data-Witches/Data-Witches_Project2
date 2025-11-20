!mkdir ~/.kaggle #create the .kaggle folder in your root directory
!echo '{"username":"stefzeemering","key":"5b2e65fcf94d620429eee3e27e514fbe"}' > ~/.kaggle/kaggle.json #write kaggle API credentials to kaggle.json
!chmod 600 ~/.kaggle/kaggle.json  # set permissions
!kaggle datasets download -d {'stefzeemering/atrial-fibrillation-classification'} -p /content/kaggle/ --force