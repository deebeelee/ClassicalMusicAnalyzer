if [ "$#" -le 2 ]
then
    echo "usage: <Composer> <filename> <shortened filename> [num_seconds]"
    exit 1 
fi
#if [ "$#" -eq 3 ]
#then
#    $num_seconds = 10
#else
#    $num_seconds = $4
#fi
newfilename="$1/$1_$3_%03d.opus"
#echo "Removing silence..."
#ffmpeg -loglevel 16 -i "$2" -af silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-50dB "$2"
echo "Splitting file (default is 10 seconds)..."
ffmpeg -loglevel 16 -i "$2" -f segment -segment_time 10 -c copy "$newfilename"
exit 0


#
# ffmpeg -i <input file> -acodec copy -ss <start time> -to <end time> <output file>
#
