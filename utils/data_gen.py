import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_customers=1000, months=12):
    """
    Generates synthetic transaction-level data for CreditLens AI with fairness variables.
    Features: customer_id, transaction_date, transaction_amount, transaction_type, employment_type, gender, age_group, loan_default
    """
    np.random.seed(42)
    random.seed(42)
    start_date = datetime.now() - timedelta(days=30 * months)
    
    customers = []
    
    # 3 types of employment: gig, salaried, self-employed
    employment_types = ['gig'] * int(num_customers * 0.5) + ['salaried'] * int(num_customers * 0.3) + ['self-employed'] * int(num_customers * 0.2)
    np.random.shuffle(employment_types)
    
    # Fairness demographics
    genders = ['Male', 'Female', 'Non-Binary']
    age_groups = ['18-25', '26-40', '41-60', '60+']
    
    transactions = []
    customer_metadata = []
    
    for i in range(num_customers):
        customer_id = f"CUST_{i:04d}"
        emp_type = employment_types[i]
        
        # Demographics
        gender = np.random.choice(genders, p=[0.55, 0.40, 0.05])
        age_group = np.random.choice(age_groups, p=[0.2, 0.5, 0.25, 0.05])
        
        # Determine base behavior based on employment type
        if emp_type == 'salaried':
            monthly_income_mean = np.random.uniform(30000, 100000)
            income_volatility = np.random.uniform(0.01, 0.05)
            monthly_tx_count = np.random.randint(10, 30)
            propensity_to_default = 0.05
        elif emp_type == 'gig':
            monthly_income_mean = np.random.uniform(15000, 50000)
            income_volatility = np.random.uniform(0.2, 0.5)
            monthly_tx_count = np.random.randint(20, 60)
            propensity_to_default = 0.15
        else: # self-employed
            monthly_income_mean = np.random.uniform(20000, 150000)
            income_volatility = np.random.uniform(0.1, 0.4)
            monthly_tx_count = np.random.randint(5, 40)
            propensity_to_default = 0.10
            
        # Determine default status upfront
        is_default = np.random.choice([0, 1], p=[1-propensity_to_default, propensity_to_default])
        
        customer_metadata.append({
            'customer_id': customer_id,
            'employment_type': emp_type,
            'gender': gender,
            'age_group': age_group,
            'loan_default': is_default
        })
        
        # Generate transactions for each month
        for month in range(months):
            # Calculate month's income
            current_month_income = max(5000, np.random.normal(monthly_income_mean, monthly_income_mean * income_volatility))
            
            # Date offset for the month
            month_start = start_date + timedelta(days=30 * month)
            
            # Income transactions (1 for salaried, multiple for gig/self-employed)
            income_tx_count = 1 if emp_type == 'salaried' else np.random.randint(2, 6)
            income_amounts = np.random.dirichlet(np.ones(income_tx_count)) * current_month_income
            
            for amt in income_amounts:
                tx_date = month_start + timedelta(days=np.random.randint(0, 28))
                transactions.append([
                    customer_id, tx_date.strftime("%Y-%m-%d"), round(amt, 2), 'income'
                ])
                
            # Expense transactions
            # Higher expenses relative to income might indicate higher default risk
            expense_ratio = np.random.uniform(0.7, 1.1) if is_default else np.random.uniform(0.4, 0.8)
            current_month_expense = current_month_income * expense_ratio
            
            expense_tx_count = max(1, monthly_tx_count - income_tx_count)
            expense_amounts = np.random.dirichlet(np.ones(expense_tx_count)) * current_month_expense
            
            for amt in expense_amounts:
                tx_date = month_start + timedelta(days=np.random.randint(0, 28))
                transactions.append([
                    customer_id, tx_date.strftime("%Y-%m-%d"), round(amt, 2), 'expense'
                ])

    df_tx = pd.DataFrame(transactions, columns=['customer_id', 'transaction_date', 'transaction_amount', 'transaction_type'])
    df_tx['transaction_date'] = pd.to_datetime(df_tx['transaction_date'])
    df_tx = df_tx.sort_values(by=['customer_id', 'transaction_date']).reset_index(drop=True)
    
    df_meta = pd.DataFrame(customer_metadata)
    
    # Merge for final dataset
    df_final = df_tx.merge(df_meta, on='customer_id', how='left')
    return df_final

if __name__ == "__main__":
    df = generate_synthetic_data(100, 6)
    print(df.head())
    df.to_csv('data/synthetic_transactions.csv', index=False)
