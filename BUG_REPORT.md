# Bug Report - Flightmare RL Training

Analysis Date: 2026-01-08
Files Analyzed: train.py, drqv2.py, wrapper.py, quadrotor_dynamics.cpp, quadrotor_env.cpp, quadrotor.cpp

---

## CRITICAL BUGS

### 1. **Warmup Thrust Too High** (wrapper.py:62-66)
**Severity:** HIGH
**Location:** `flightrl/drqv2_src/wrapper.py:62-66`

**Issue:**
```python
warmup_action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
# ...
remapped[0, 0] = (warmup_action[0, 0] + 1.0) / 2.0  # Maps to 0.5
```

The warmup action of 0.0 in [-1, 1] range remaps to 0.5 in [0, 1] range, giving:
- Thrust = 0.5 × 27.07 m/s² = **13.5 m/s²**
- Hover thrust needed = **9.81 m/s²**
- **Result: Drone accelerates upward during warmup!**

**Fix:**
```python
# For hover: need action ≈ -0.28 in [-1,1] range
warmup_action = np.array([[-0.28, 0.0, 0.0, 0.0]], dtype=np.float32)
```

---

### 2. **Division by Zero Risk in Gate Pass Detection** (wrapper.py:276)
**Severity:** MEDIUM
**Location:** `flightrl/drqv2_src/wrapper.py:276`

**Issue:**
```python
total_dist_change = dist_curr - dist_prev
t = (0 - dist_prev) / total_dist_change  # No check for zero!
```

If the drone barely moves between steps, `total_dist_change` could be ~0, causing:
- Division by zero
- Numerical instability
- Incorrect intersection calculations

**Fix:**
```python
total_dist_change = dist_curr - dist_prev
if abs(total_dist_change) < 1e-6:
    return False  # Not moving significantly
t = -dist_prev / total_dist_change
```

---

### 3. **Confusing Double Negative Logic** (wrapper.py:273)
**Severity:** LOW (Logic Bug)
**Location:** `flightrl/drqv2_src/wrapper.py:273`

**Issue:**
```python
if not (np.sign(dist_prev) != np.sign(dist_curr)):
    return False
```

Double negative `not (...!=...)` is equivalent to `==` but unnecessarily confusing.

**Fix:**
```python
# Check if signs are the same (didn't cross plane)
if np.sign(dist_prev) == np.sign(dist_curr):
    return False
```

---

## DESIGN ISSUES

### 4. **Perception Reward Exploitable** (wrapper.py:250)
**Severity:** MEDIUM
**Location:** `flightrl/drqv2_src/wrapper.py:250`

**Issue:**
```python
if dist_to_gate < 0.2: return 1.0, 1.0
```

When drone is within 0.2m of gate, maximum perception reward is given **without checking camera direction**. Agent could exploit this by just flying close to gates without looking at them.

**Recommendation:**
Remove the early return or significantly reduce the distance threshold (e.g., 0.05m).

---

### 5. **Exception Handling Masks Real Errors** (train.py:229-233)
**Severity:** MEDIUM
**Location:** `flightrl/drqv2_src/train.py:229-233`

**Issue:**
```python
except RuntimeError as e:
    print(f"[Python] ⚠️ Environment Error: {e}")
    next_obs = env.reset()
    reward = 0.0
    done = True
```

Catching `RuntimeError` and silently resetting makes debugging difficult. Real bugs in C++ code might be hidden.

**Recommendation:**
```python
except RuntimeError as e:
    print(f"[Python] ❌ CRITICAL Environment Error: {e}")
    print(f"[Python] State: obs shape={obs.shape}, action={action}")
    # Save debug info before resetting
    import traceback
    traceback.print_exc()
    next_obs = env.reset()
    reward = 0.0
    done = True
```

---

### 6. **Hover Thrust Not Centered in Action Space** (Design Choice)
**Severity:** LOW
**Location:** `wrapper.py:34-40`, `quadrotor_env.cpp:48-59`

**Issue:**
The action remapping makes hover thrust (~0.36 in [0,1] range) correspond to -0.28 in [-1,1] range. This is asymmetric and might make learning harder.

**Current Mapping:**
- DrQv2 output: -1.0 → Thrust: 0 m/s² (freefall)
- DrQv2 output: **-0.28** → Thrust: 9.81 m/s² (hover)
- DrQv2 output: +1.0 → Thrust: 27.07 m/s² (max)

**Alternative Design:**
Center the action space around hover:
- Map [-1, 1] → [0, 2×hover_thrust]
- This makes 0 correspond to hover, simplifying learning

**Code Change:**
```python
# In wrapper.py
remapped_action[0] = (action[0] + 1.0) * hover_thrust  # Centers around hover
```

---

## POTENTIAL ISSUES

### 7. **No Validation of C++ State Values** (wrapper.py:140-150)
**Severity:** LOW
**Location:** `flightrl/drqv2_src/wrapper.py:140-150`

**Issue:**
No validation that C++ returns finite values for position, velocity, quaternions.

**Recommendation:**
```python
if not np.all(np.isfinite(drone_pos)) or not np.all(np.isfinite(drone_vel)):
    print(f"[Wrapper] Warning: Non-finite state! Pos={drone_pos}, Vel={drone_vel}")
    # Handle error appropriately
```

---

### 8. **Gate Collision Detection May Have False Positives** (quadrotor_env.cpp:360-409)
**Severity:** LOW
**Location:** `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp:360-409`

**Issue:**
The oriented bounding box (OBB) collision detection uses fixed thresholds:
```cpp
const Scalar DRONE_RADIUS = 0.1;  // 10cm radius
const Scalar OUTER_FRAME_HALF_SIZE = 1.15;  // Gate outer size
```

If the actual drone size (set via `setSize()`) doesn't match `DRONE_RADIUS`, collisions may be incorrectly detected.

**Verification Needed:**
Check that `quadrotor_ptr_->setSize(quad_size)` at line 44 matches the collision radius.
Currently: `quad_size(0.2, 0.2, 0.2)` but collision uses radius 0.1 (diameter 0.2). This is **consistent**.

---

### 9. **Motor Dynamics Time Constant Discrepancy** (quadrotor_dynamics.cpp:24 vs config)
**Severity:** LOW
**Location:** Hard-coded vs YAML config

**Issue:**
In constructor (line 24):
```cpp
motor_tau_inv_ = 1.0 / 0.033;  // τ = 33ms
```

In updateParams (line 233):
```cpp
motor_tau_inv_ = (1.0 / params["quadrotor_dynamics"]["motor_tau"].as<Scalar>());
```

From config: `motor_tau: 0.0001` → τ = 0.1ms (much faster)

**Impact:** The config value (0.1ms) overrides the constructor value (33ms). Motors respond almost instantaneously, which may not be realistic but is acceptable for RL.

---

### 10. **Incomplete Debug Counter in Step Function** (wrapper.py:131-135)
**Severity:** TRIVIAL
**Location:** `flightrl/drqv2_src/wrapper.py:131-135`

**Issue:**
```python
if not hasattr(self, '_step_count'):
    self._step_count = 0
if self._step_count < 5:
    print(f"[Step {self._step_count}] ...")
self._step_count += 1
```

The counter is never reset between episodes, so debug prints only appear in the first episode.

**Fix:**
Reset in `reset()`:
```python
def reset(self):
    self._step_count = 0  # Add this
    # ... rest of reset
```

---

## OBSERVATIONS (Not Bugs)

### 11. **Frame Stacking Only During Video Episodes**
**Location:** `train.py:186-190`

Frames are only captured every `VIDEO_INTERVAL` (500) episodes. This is intentional to save memory but could be confusing if you expect videos from every episode.

### 12. **Quaternion Convention Consistency**
**Confirmed:** Both C++ and Python use **[x, y, z, w]** convention consistently.
- C++ (quadrotor_env.cpp:159): `Vector<4> q_vec(q.x(), q.y(), q.z(), q.w());`
- Python (wrapper.py:157): `drone_quat = np.array([qx, qy, qz, qw])`

### 13. **NaN Used as Sentinel Value**
**Location:** `quadrotor_env.cpp:193`
```cpp
cmd_.thrusts.setConstant(NAN);
```
This forces the use of high-level controller (`runFlightCtl`) instead of direct motor control. It's a hack but works correctly.

---

## RECOMMENDED FIXES PRIORITY

**Immediate (Critical):**
1. Fix warmup thrust calculation
2. Add division-by-zero check in gate pass detection

**Short Term (Important):**
3. Simplify double negative logic
4. Improve exception handling with better logging
5. Consider redesigning action space to center on hover

**Long Term (Nice to Have):**
6. Add state validation
7. Reset debug counter between episodes
8. Review perception reward distance threshold

---

## TESTING RECOMMENDATIONS

1. **Warmup Behavior:** Monitor drone position during `connect_and_warmup()` - should stay ~stationary
2. **Gate Passing:** Test with slow-moving drone near gate boundaries to verify no division errors
3. **Hover Stability:** Verify agent can learn stable hovering with current action mapping
4. **Collision Detection:** Validate that collision thresholds match actual drone size
