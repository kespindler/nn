.PHONY:run
run:
	. env/bin/activate && ipython -i nn.py


data/x.pkl:
	curl -o data/test.csv "https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3004/test.csv?sv=2012-02-12&se=2016-04-01T00%3A48%3A26Z&sr=b&sp=r&sig=unQTloJ7cHqiWOymdZTzw9vUra9JmeyOjUEpQjIRbM4%3D"
	curl -o data/train.csv "https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3004/train.csv?sv=2012-02-12&se=2016-04-01T00%3A48%3A22Z&sr=b&sp=r&sig=GeWug4DxjnEoBsFTf6Iift5lT%2FlK1GR1WbmNuXdKd0g%3D"
	python -c 'import nn;nn.save_from_csv()'


data: data/x.pkl
	echo Hi


w:
	python -c 'import nn;nn.save_new_w()'


bootstrap:
	virtualenv env
	. env/bin/activate && pip install Pillow ipdb numpy


openvis:
	open vis

.DEFAULT:run
