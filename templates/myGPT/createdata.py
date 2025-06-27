import os
import pandas as pd
import numpy as np
import random

# Create 'data' directory if not exists
os.makedirs('./data', exist_ok=True)

# Generate synthetic dataset
num_samples = 100
patient_ids = range(1, num_samples + 1)
ages = np.random.randint(18, 65, size=num_samples)
methods = ['intravenous', 'subcutaneous', 'intramuscular']
aso_sequences = [''.join(random.choices(['A','T','C','G'], k=15)) for _ in range(num_samples)]
dosages = np.round(np.random.uniform(5, 50, size=num_samples), 2)  # mg
baseline_levels = np.round(np.random.uniform(5, 30, size=num_samples), 2)  # % of normal clotting factor
improvements = np.round(np.random.uniform(10, 50, size=num_samples), 2)
post_levels = np.minimum(baseline_levels + improvements, 100)  # cap at 100%
side_effect_scores = np.round(np.random.uniform(0, 5, size=num_samples), 2)

df = pd.DataFrame({
    'patient_id': patient_ids,
    'age': ages,
    'aso_sequence': aso_sequences,
    'delivery_method': np.random.choice(methods, size=num_samples),
    'dosage_mg': dosages,
    'baseline_factor_level_pct': baseline_levels,
    'post_treatment_factor_level_pct': post_levels,
    'improvement_pct': post_levels - baseline_levels,
    'side_effect_score': side_effect_scores
})

# Save to CSV
csv_path = './data/dataset.csv'
df.to_csv(csv_path, index=False)

