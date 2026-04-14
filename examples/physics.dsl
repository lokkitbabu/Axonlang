// ── Runge-Kutta 4th order — integrates dy/dt = f(y,t) for one step ───────────
//
// k1 = f(y, t)
// k2 = f(y + dt/2 * k1, t + dt/2)
// k3 = f(y + dt/2 * k2, t + dt/2)
// k4 = f(y + dt * k3,   t + dt)
// y_next = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
//
// Here f(y,t) = -lambda*y  (exponential decay)
fn rk4_exp_decay(y: f64, t: f64, dt: f64, lambda: f64) -> f64 {
    let k1 = -lambda * y;
    let k2 = -lambda * (y + dt * 0.5 * k1);
    let k3 = -lambda * (y + dt * 0.5 * k2);
    let k4 = -lambda * (y + dt * k3);
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ── Van der Pol oscillator — single RK4 step for x component ─────────────────
// dx/dt = v
// dv/dt = mu*(1 - x^2)*v - x
fn van_der_pol_x(x: f64, v: f64, mu: f64, dt: f64) -> f64 {
    let k1x = v;
    let k1v = mu * (1.0 - x^2.0) * v - x;
    let k2x = v + dt * 0.5 * k1v;
    let k2v = mu * (1.0 - (x + dt * 0.5 * k1x)^2.0) * (v + dt * 0.5 * k1v) - (x + dt * 0.5 * k1x);
    let k3x = v + dt * 0.5 * k2v;
    let k4x = v + dt * k2v;
    return x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
}

// ── 2-body gravitational acceleration (along x-axis) ─────────────────────────
// F = G*m1*m2/r²    a1 = F/m1 = G*m2/r²
fn grav_accel(G: f64, m2: f64, r: f64) -> f64 {
    return G * m2 / (r^2.0);
}

// ── Kepler orbit energy (specific, per unit mass) ────────────────────────────
// E = 0.5*v² - G*M/r
fn kepler_energy(v: f64, G: f64, M: f64, r: f64) -> f64 {
    return 0.5 * v^2.0 - G * M / r;
}

// ── Black-Scholes European call option price ─────────────────────────────────
// C = S*N(d1) - K*exp(-r*T)*N(d2)
// d1 = (log(S/K) + (r + sigma²/2)*T) / (sigma*sqrt(T))
// d2 = d1 - sigma*sqrt(T)
// Approximating N(x) ≈ 0.5*(1 + erf(x/sqrt(2))) via the logistic approximation
//   N(x) ≈ 1/(1+exp(-1.7*x))  (fast approximation, error < 0.004)
fn bs_d1(S: f64, K: f64, r: f64, sigma: f64, T: f64) -> f64 {
    return (log(S / K) + (r + sigma^2.0 * 0.5) * T) / (sigma * sqrt(T));
}

fn bs_call(S: f64, K: f64, r: f64, sigma: f64, T: f64) -> f64 {
    let d1    = (log(S / K) + (r + sigma^2.0 * 0.5) * T) / (sigma * sqrt(T));
    let d2    = d1 - sigma * sqrt(T);
    let nd1   = 1.0 / (1.0 + exp(-1.7 * d1));
    let nd2   = 1.0 / (1.0 + exp(-1.7 * d2));
    let disc  = exp(-r * T);
    return S * nd1 - K * disc * nd2;
}

// ── Black-Scholes delta (∂C/∂S) — should equal N(d1) ≈ nd1 above ─────────────
// Used to verify autodiff: tape_grad(bs_call, [S,K,r,σ,T])[0] ≈ N(d1)
fn bs_delta(S: f64, K: f64, r: f64, sigma: f64, T: f64) -> f64 {
    let d1  = (log(S / K) + (r + sigma^2.0 * 0.5) * T) / (sigma * sqrt(T));
    let nd1 = 1.0 / (1.0 + exp(-1.7 * d1));
    return nd1;
}

// ── Sigmoid and softplus (useful for ML activation layers) ───────────────────
fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + exp(-x));
}

fn softplus(x: f64) -> f64 {
    return log(1.0 + exp(x));
}

// ── Mean squared error loss ───────────────────────────────────────────────────
fn mse(pred: f64, target: f64) -> f64 {
    return (pred - target)^2.0;
}

// ── Cross-entropy loss (binary) ───────────────────────────────────────────────
fn bce_loss(p: f64, y: f64) -> f64 {
    return -(y * log(p) + (1.0 - y) * log(1.0 - p));
}

// ── Electrostatic potential energy between two charges ────────────────────────
fn coulomb(k_e: f64, q1: f64, q2: f64, r: f64) -> f64 {
    return k_e * q1 * q2 / r;
}

// ── Lennard-Jones 12-6 potential ──────────────────────────────────────────────
fn lennard_jones(epsilon: f64, sigma: f64, r: f64) -> f64 {
    let s_over_r = sigma / r;
    return 4.0 * epsilon * (s_over_r^12.0 - s_over_r^6.0);
}

// ── Loop: accumulated quadrature (trapezoidal rule) ───────────────────────────
// Approximates ∫₀¹ sin(x) dx ≈ 1 - cos(1) ≈ 0.4597
// n = number of intervals  (use small n for testing, e.g. n=10)
fn trapz_sin(n: f64) -> f64 {
    let h   = 1.0 / n;
    let acc = 0.0;
    for i in 0..10 {
        let x0 = i * h;
        let x1 = (i + 1.0) * h;
        let acc = acc + 0.5 * h * (sin(x0) + sin(x1));
    }
    return acc;
}
