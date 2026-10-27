## Controller

`BaselinePolicy` is the first submitted controller: the organizers' PID with
per-zone comfort bands read from sensors and outside temperature decoded from
its normalized observation before the economizer decision. `CoolingHoldPolicy`
adds the cooling-run hold and, when enabled, the night cooling ceiling. Actuator indices come from each building's metadata,
so the controller adapts to any number of zones and AHUs.
