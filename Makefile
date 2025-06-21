FILE_ID = 1tAovZQKz496qwRGnJQVBZvaSlTqvEvxy
ZIP_FILE = dataset.zip
DATA_DIR = data
TEMP_DIR = dataset

.PHONY:	download unzip reorganize clean

all:	download unzip reorganize clean 

download:
	gdown https://drive.google.com/uc?id=$(FILE_ID) -O $(ZIP_FILE)

unzip: download
	unzip -o $(ZIP_FILE) -d $(DATA_DIR)

reorganize:	unzip
	mv $(DATA_DIR)/$(TEMP_DIR)/train $(DATA_DIR)/
	mv $(DATA_DIR)/$(TEMP_DIR)/test $(DATA_DIR)/
	rm -rf $(DATA_DIR)/$(TEMP_DIR)

clean:
	rm -rf $(ZIP_FILE)



