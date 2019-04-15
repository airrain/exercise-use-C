void schedule(){
	int i,j,k,msx = 0;
	for(i = 0;i < m ;i++){
		d[i] = 0;
		for(j = 0;j < n;j++)
			s[i][j] = 0;
	}
}
	for(i = 0;i < m;i++){
		s[i][0] = i;
		d[i] = d[i] + t[i];
		count[i] = 1;
	}
	for(j = 1;j < m;j++){
		int min = d[0];
		k = 0;
		for(j = 1;j < m;j++){
			if(min > d[j]){
				min = d[j];
				k = j;
			}
		}
		s[k][0] = i;
		count[k] = count[k] + 1;
		d[k] = d[k] + t[i];
		for(i = 0;i < m;i++){
			if(max < d[i]){
				max = d[i];
			}
			
		}
		
		
		
		
		
		
		
		
		
		
		
	}
