MAIN=main.py

run:
	rm -rf images
	mkdir -p images/hist
	mkdir -p images/violin
	mkdir -p images/boxplot
	python3 $(MAIN)

generate:
	python3 data_load.py

plot:
	python3 plot.py

print:
	python3 print.py

statistics:
	python3 descriptive.py

random:
	python3 missing_values.py

git:
	git pull --no-rebase
	git add .
	read -p "Add message: " input; git commit -m "$$input"
	git push

clean:
	rm -rf __pycache__/
	rm -rf images