ND:
0. RNN NO Gating [128, 256, 512] new loss + bn
1. RNN NO Gating [128, 256, 512] new loss early stopping 3 + bn
2. 2-stack RNN NO Gating [256] new loss early stopping 3  + bn
3. 3-stack RNN NO Gating [256,512] new loss early stopping 3 + bn
4. 2-stack RNN NO Gating [256,512, 1024] new loss early stopping 3  + bn

A:
0. [512,1024] + gating + es + NDF
1. [512,1024] + gating + es + NDF + bn
2. [256,512,1024] + gating + es + NDF
3. [256,512,1024] + gating + es + NDF + bn
4.

S:
0. [512,1024] + gating + es + NDF
1. [512,1024] + gating + es + NDF + bn
2. [256,512,1024] + gating + es + NDF
3. [256,512,1024] + gating + es + NDF + bn
4.

E:
0. [512,1024] + gating + NDF
1. [512,1024] + gating + NDF + bn
2. [256,512,1024] + gating + NDF
3. [256,512,1024] + gating + NDF + bn