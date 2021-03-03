#docker run --rm -it -v /mnt/au_dfs/ST_FC-WP1/:/data/ jhj/sentinelproc \
docker run --rm -it -v $PWD:/data/ jacobjeppesen/sentinelproc \
	--username username \
	--password password \
	--data_directory /data/sentinelproc \
	--geojson denmark_without_bornholm \
	--satellite s1 \
	--startdate 20190620 \
	--enddate 20190621 \
	--s1_num_proc 2 \
	--s1_del_intermediate True \
	--s1_output_crs EPSG:32632 \
	--s2_num_proc 6 \
	--s2_relative_orbit 0
