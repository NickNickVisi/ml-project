# commit.sh
#!/bin/bash

read -p "Choose which part to run: " input
if [ "$input" == "1" ]; then
    echo "Running part 1"
    cd ParteaI
    make	
    cd ..
elif [ "$input" == "2" ]; then
    echo "Running part 2"
    cd ParteaII
    make 	
    cd ..
else
    echo "Invalid input. Please enter 1 or 2."
fi
