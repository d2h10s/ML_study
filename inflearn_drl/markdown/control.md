# type2의 제어 시스템의 해석
오늘 우리는 type2 제어 시스템에서 $M_p, t_s, t_r$을 구하는 방정식이 어떻게 도출되는지 알아 볼 것이다.  
우선 시스템에 대한 전달함수를 구하고 역 라플라스를 취하여 시간에 대한 방정식으로 바꾸는 것이 시작이다.  
$0 \leq \zeta < 1$인 under damped 시스템이라고 가정할 때 전달함수 $G(s)$는 아래와 같다.
$$G(s)=\frac{Y(s)}{R(s)}=\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}$$
여기서 $R(s)$가 unit step function, 즉 $\frac1s$일 때 $Y$에 대해 정리하면 다음과 같다.
$$Y(s)=\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}\frac1s$$
헤비사이드 부분 분수 분리법을 이용하면 $\frac1s$ 는 쉽게 분리가 된다.  
분모의 차수에 따라 미지의 상수를 정하고 분모가 0이 되는 항의 분모를 양 변에 곱한다.
$$Y(s)=(\frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}\frac1s)\times s = (\frac As+\frac{Bs+C}{s^2+2\zeta\omega_n s+\omega_n^2)})\times s$$
여기서 $s = 0$을 대입하면
$\frac{\omega_n^2}{\omega_n^2}=A$ 가 되므로, $A = 1$ 이 된다.  
$$Y(s)=\frac 1s+\frac{Bs+C}{s^2+2\zeta\omega_n s+\omega_n^2}$$
나머지 $B, C$는 통분을 하면 쉽게 구할 수 있다.  
통분을 먼저 시도 해보자.
$$ \frac{\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2 }\times \frac1s = \frac{Bs^2+Cs+s^2+2\zeta\omega_n s+\omega_n^2}{s^2+2\zeta\omega_n s+\omega_n^2}\times \frac1s$$
양변은 항등식이므로 분자에서 $B,C$가 있는 $s^2$과 $s$항을 살펴보면 해답을 찾을 수 있다.
$$Bs^2+s^2=0 \Rightarrow B = -1$$
$$ Cs + 2\zeta\omega_ns = 0 \Rightarrow C = -2\zeta\omega_n $$
그러면 다음과 같이 정리할 수 있다.

$$Y(s)=\frac 1s-\frac{s+2\zeta\omega_n}{s^2+2\zeta\omega_n s+\omega_n^2}$$
우변 두번째 항을 보면 분모가 복잡한 2차식이기 때문에 세 가지 라플라스 역변환 공식을 이용할 것이다.
1. $e^{at}=\mathcal{L}^{-1}\{\frac1{s-a}\}$  
2. $sin(at)=\mathcal{L}^{-1}\{\frac a{s^2-a^2}\}$
3. $cos(at)=\mathcal{L}^{-1}\{\frac s{s^2-a^2}\}$  

우선 위의 공식을 이용할 수 있도록 분모를 정리해보자.
$$Y(s)=\frac 1s-\frac{s+2\zeta\omega_n}{(s+\zeta\omega_n)^2-\zeta^2\omega_n^2 + \omega_n^2}$$
$$Y(s)=\frac 1s-\frac{s+2\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_n^2(1-\zeta^2)}$$
$$Y(s)=\frac 1s-\frac{s+2\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_d^2}(\because \omega_d=\sqrt{1-\zeta^2})$$
이제 거의 다 된 것 같다. 공식에 맞게 분수를 분리하여 보자.
$$Y(s)=\frac 1s-\frac{s+\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_d^2}+\frac{\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_d^2}$$
이제 라플라스 변환을 할 수 있을 것 같다.
$$\mathcal{L}^{-1}\{ Y(s)\}=\mathcal{L}^{-1}\{\frac 1s-\frac{s+\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_d^2}+\frac{\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_d^2}\}$$
$$y(t)=\mathcal{L}^{-1}\{\frac 1s\}-\mathcal{L}^{-1}\{\frac{s+\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_d^2}\}+\mathcal{L}^{-1}\{\frac{\zeta\omega_n}{(s+\zeta\omega_n)^2+\omega_d^2}\}$$
1번 공식을 적용하여 역 라플라스 s에 붙어있는 $\zeta\omega_n$을 빼내자.
$$y(t)=1-e^{-\zeta\omega_n}\mathcal{L}^{-1}\{\frac{s}{s^2+\omega_d^2}\}+e^{-\zeta\omega_n}\mathcal{L}^{-1}\{\frac{\zeta\omega_n}{s^2+\omega_d^2}\}$$
이제 2번 공식을 이용하여 두번째 항을 변환하면 다음과 같다.
$$y(t) = 1-e^{-\zeta\omega_n}(cos(\omega_dt)+\mathcal{L}^{-1}\{\frac{\zeta\omega_n}{s^2+\omega_d^2}\})$$
세 번째 항은 바로 변환할 수 없기 때문에 조금 변형이 필요하다.
$$\omega_d=\sqrt{1-\zeta^2}$$
위 식을 사용하면 $\omega_n$을 $\omega_d$로 바꿀 수 있을 것 같다.
$$y(t) = 1-e^{-\zeta\omega_n}(cos(\omega_dt)+\mathcal{L}^{-1}\{\frac{\zeta}{s^2+\omega_d^2}\frac{\omega_d}{\sqrt{1-\zeta^2}}\})$$
상수를 정리하면 다음과 같다.
$$y(t) = 1-e^{-\zeta\omega_n}(cos(\omega_dt)+\frac{\zeta}{\sqrt{1-\zeta^2}}\mathcal{L}^{-1}\{\frac{\omega_d}{s^2+\omega_d^2}\})$$
따라서 완전한 변환식을 얻을 수 있다.
$$y(t) = 1-e^{-\zeta\omega_n}(cos(\omega_dt)+\frac{\zeta}{\sqrt{1-\zeta^2}}sin(\omega_dt))$$
놀랍게도 우리는 한 번 더 정리할 수 있다.
오래 전 학창 시절에 배웠던 삼각함수의 합성 공식을 이용하자.
$$ a sin x + b cos x=\sqrt{a^2+b^2}sin(x+\alpha)$$
위 식을 이용하여 정리하자, 여기서 우리는 $\alpha$ 값은 궁금하지 않기 때문에 미지수로 남겨두자.
$$ y(t) = 1-\frac{e^{-\zeta\omega_n}}{\sqrt{1-\zeta^2}} sin(\omega_dt+\Phi)$$
그럼 우리는 2차 시스템의 라플라스 변환을 마쳤다.  
이제야 우리가 다루는 시스템의 그래프가 왜 출렁이는지 알 수 있게 되었다.
두번째 항이 시간이 지남에 따라서 요동치기 때문이다.