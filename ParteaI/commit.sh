# commit.sh
#!/bin/bash

read -p "Add message: " input
git commit -m "$input"