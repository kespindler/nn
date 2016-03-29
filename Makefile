.PHONY:run
run:
	. env/bin/activate && ipython -i nn.py

.DEFAULT:run
