prepare:
	sudo pip install virtualenv; \
	mkdir ~/virt_env; \
	virtualenv virt_env/virt1; \
	source ~/virt_env/virt1/bin/activate; \
	cd lib/pp-1.6.4/ && python setup.py install && cd ../..; \
	sudo pip install -r requirements.txt; \

run:
	source ~/virt_env/virt1/bin/activate; \
	cd src; \
	python spellCheck.py ${ARGS}; \

run_server:
	source ~/virt_env/virt1/bin/activate; \
	~/virt_env/virt1/bin/ppserver.py -a -d	
