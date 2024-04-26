#!/bin/zsh

# DPI     : 20000 15000 10000  5000
# POLL    :  8000  4000  2000 
# VOLUME  :    80    70    60    50

function sigterm_handler() {
  # happens at 7am until 8pm = 46800 seconds
  echo "waiting for interval -- sleeping from 7am to 8pm - schedule_test"
  current_epoch=$(date +%s)
  target_epoch=$(date -d 'today 20:00' +%s)
  #target_epoch=$current_epoch
  #target_epoch=$(date  "+%s" -d "1 second")
  sleep_seconds=$(( $target_epoch - $current_epoch ))


  if [[ $sleep_seconds -ge 0 ]]
  then
    echo "sleeping for $sleep_seconds seconds"
    sleep $sleep_seconds | pv -t
  fi
}

#trap sigterm_handler SIGTERM
echo "$$" > c_pid


touch c_test
touch c_spkr

export realdb=0


function get_test_id () {

  # USAGE

  if [[ $# -ne 3 ]]
  then
    echo "usage: [pollrate] [dpi] [db]"
    echo "  | this function will return the test id to place in ./c_test"
    echo "  | and also set the correct parameters for running the specified test"
    echo "  | and also move the correct directories"
    echo "  | progress will be saved in case of interruption in both"
    echo "    | c_test <-- test configuration"
    echo "    | c_spkr <-- position inside the test in c_test"
    exit 1 
  fi

  if [[ "$3" == "50" ]] 
  then
    realdb=80 
  elif [[ "$3" == "40" ]] 
  then
    realdb=70 
  elif [[ "$3" == "30" ]] 
  then
    realdb=60 
  elif [[ "$3" == "20" ]] 
  then
    realdb=50 
  else
    echo "bad volume scale"
    exit 1
  fi

  razer-cli --poll $1
  razer-cli --dpi $2
  # realdb stores the volume parameter to pass to the script 

  echo "$1_$2_$realdb"

}

function run_test() {

  # takes one argument - test_id, must run get_test_id first
  # runs with the preloaded settings from get_test_id
  echo "running test: $1"
  echo "$1" > ./c_test
  v_level="${1: -2}"
  mkdir -p gen/csv
  ./create.sh 0 NONE $v_level
  rm -rf result/$1
  mv gen/csv result/$1

  return 100
}

test_flag=$(<c_test)
test_ok=0
echo "stored test flag: $test_flag"

#
##
### BLOCK 1 - 80dB
##
#

n_test="$(get_test_id 8000 20000 50)"
if [[ "$test_flag" == "" || "$test_flag" == "$n_test" ]]
then
  run_test "$n_test"
  test_ok=$?
fi
n_test="$(get_test_id 8000 15000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
n_test="$(get_test_id 8000 10000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
n_test="$(get_test_id 8000 5000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi

n_test="$(get_test_id 4000 20000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
n_test="$(get_test_id 4000 15000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
n_test="$(get_test_id 4000 10000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
n_test="$(get_test_id 4000 5000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi

n_test="$(get_test_id 2000 20000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
n_test="$(get_test_id 2000 15000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
n_test="$(get_test_id 2000 10000 50)"
if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
then
  run_test "$n_test"
	test_ok=$?
fi
#n_test="$(get_test_id 2000 5000 50)"
#if [[ "$test_flag" == "$n_test" || $test_ok -eq 100 ]]
#then
  #run_test "$n_test"
	#test_ok=$?
#fi

#
##
### BLOCK 2 - 70dB
##
#

n_test="$(get_test_id 8000 20000 40)"
if [[ "$test_flag" == "" ]]
then
  run_test "$n_test"
  test_ok=$?
fi

#
##
### BLOCK 3 - 60dB
##
#

n_test="$(get_test_id 8000 20000 30)"
if [[ "$test_flag" == "" ]]
then
  run_test "$n_test"
  test_ok=$?
fi

##
### BLOCK 4 - 50dB
##
#

n_test="$(get_test_id 8000 20000 20)"
if [[ "$test_flag" == "" ]]
then
  run_test "$n_test"
  test_ok=$?
fi
#else 
  #echo "failed to match test, delete ./c_test and run script again"
  #exit 1
#fi


## static parameters above
## dynamic parameters must be changed on each run
#razer-cli --dpi 15000
#./create.sh 0 NONE 50
#mv gen/csv result/8khz_15k_80db
#razer-cli --dpi 10000
#./create.sh 0 NONE 50
#mv gen/csv result/8khz_10k_80db
#razer-cli --dpi 5000
#./create.sh 0 NONE 50
#mv gen/csv result/8khz_5k_80db
