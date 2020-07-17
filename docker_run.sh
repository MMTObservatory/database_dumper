#!/bin/bash
sudo docker run -p 8000:3000 -v $(pwd)/tmp:/tmp -v $(pwd)/user_data:/user_data -v $(pwd)/config:/config -e "DB_DUMPER_CONFIG=/config/config.ini" -v $(pwd):/db_dumper -it db_dumper 
