#define N 100
int cost[N][N];
int trace[N][N];
int cmm(int n,int seq[]){
	int tempCost;
	int tempTrace;
	int i,j,k,p;
	int temp;
	for(i = 0;i < n;i++){
		cost[i][i] = 0;
	}
	for(p = 1;p < n;p++){
		for(i = 0;i < n-p;i++){
			j = n + P;
			tempCost = -1;
			for(k = i;k < j;k++){
				temp = cost[i][k] + cost[k+1][j] + seq[i] * seq[k+1] * seq[j+1]
				if(tempCost = -1 || tempCost > temp){
					tempCost = temp;
					tempTrace = k;
				}
			}
			cost[i][j] = tempCost;
			trace[i][j] = tempTrace;
		}
	}
	return cost[0][n-1]; 
	}
}
