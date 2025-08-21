import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
Customers = [
     "Arab Tunisian Bank (ATB)",
    "Banque Nationale Agricole (BNA)", "Attijari Bank", "Banque de Tunisie (BT)", 
    "Amen Bank (AB)", "Banque Internationale Arabe de Tunisie (BIAT)", 
   "La Poste Tunisienne",
     "Banque de l'Habitat (BH)", 
    "Arab Banking Corporation (ABC)",
    "Banque Tuniso-Libyenne (BTL)"
]
gen = 10000
np.random.seed(0)
random.seed(0)
start_of_year = datetime(2024, 1, 1)
delta = datetime.now() - start_of_year
data = {
    'Customers': np.random.choice(Customers, gen),
    'Quantity': np.random.randint(20, 200, gen),
    'Date': [datetime.now().date() - timedelta(days=random.randint(delta.days, 8766)) for _ in range(gen)]
}
unit_prices = np.random.randint(10000, 20000, gen)
data['Total Payment in TND'] = data['Quantity'] * unit_prices
df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)
