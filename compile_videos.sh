rm -rf Videos/*
for i in {000..007}; do
  ffmpeg -framerate 30 -start_number 0 -i Frames/ep${i}/frame%04d.png -frames:v 70 -c:v libx264 -pix_fmt yuv420p Videos/hammer_ep${i}.mp4
done
