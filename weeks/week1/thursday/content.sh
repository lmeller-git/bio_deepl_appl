mkdir -p ~/all_seqs
%cd ~/
wget -P ~/all_seqs/ https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz
gzip -df "all_seqs/Homo_sapiens.GRCh38.cds.all.fa.gz"
