# ss format name is: ss<epnum><screencapnum> where:
#   the last 5 digits are the screencap number
#   the remaining leading digits are the episode number starting from 1
savedir="e:/precure/smile_precure_ss"

if [ \( "$#" -gt 2 \) -o \( "$#" -eq 0 \) ]; then
  echo "usage: $0 <all | <filename ep#>>"
  exit 1
fi

if [ "$1" != "all" ]; then
  echo "extracting one ep..."

  cmd="ffmpeg -threads 6 -i '$1' -qscale:v 2 -f image2 -vf fps=fps=1 '$savedir/ss$2%5d.jpg'"
  echo $cmd
  eval $cmd
  exit 1
fi

echo "extracting all eps..."
cnt=0

# get file list of only mp4s and mkvs
shopt -s nullglob
vids=(*.{mp4,mkv})
shopt -u nullglob

for vid in "${vids[@]}"
do
  cnt=$[$cnt+1]

  cmd="ffmpeg -threads 6 -i '$vid' -qscale:v 2 -f image2 -vf fps=fps=1 '$savedir/ss$cnt%5d.jpg'"
  echo $cmd
  eval $cmd
done
