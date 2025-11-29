import controlgym
from controlgym.envs.linear_control import LinearControlEnv
import os

# Try to find package path
package_path = os.path.dirname(controlgym.__file__)
print(f"Package path: {package_path}")

# List files in package to find data
for root, dirs, files in os.walk(package_path):
    for file in files:
        if file.endswith('.mat') or file.endswith('.json'):
            print(os.path.join(root, file))

# Try to instantiate
try:
    env = LinearControlEnv(id='msd')
    print("Successfully instantiated LinearControlEnv(id='msd')")
except Exception as e:
    print(f"Failed to instantiate LinearControlEnv(id='msd'): {e}")

try:
    env = LinearControlEnv(id='MassSpringDamper')
    print("Successfully instantiated LinearControlEnv(id='MassSpringDamper')")
except Exception as e:
    print(f"Failed to instantiate LinearControlEnv(id='MassSpringDamper'): {e}")
