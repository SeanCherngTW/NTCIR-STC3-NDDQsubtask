ND & DQ
DQ:
A: 3stack CNN & 2stack CNN gating+es
S: 3stack CNN & 2stack CNN gating+es
E: 3stack CNN gating

0. 1-stack RNN NO Gating [256] new loss & A1S1E
1. 2-stack RNN NO Gating [256,512] new loss & A1S1E
2. 3-stack RNN NO Gating [256,512,1024] new loss & A1S1E
3. 2-stack RNN NO Gating [128] new loss early stopping 3 & A2S2E
4. 3-stack RNN NO Gating [128, 256] new loss early stopping 3  & A2S2E
5. 1-stack RNN NO Gating [128, 256, 512] new loss early stopping 3  & A2S2E
6. 2-stack RNN NO Gating [256] new loss early stopping 3  & A2S2E
7. 3-stack RNN NO Gating [256,512] new loss early stopping 3  & A2S2E
8. 2-stack RNN NO Gating [256,512, 1024] new loss early stopping 3  & A2S2E
