all:
	python3 setup.py build_ext --inplace
clean:
	rm -rfv build *.pyc src/*.so src/*.c src/*.pyc
