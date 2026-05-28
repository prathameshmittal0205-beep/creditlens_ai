import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_data(num_customers=500, months=12):
    """
    Generates realistic synthetic transaction-level data for CreditLens AI.
    Features: customer_id, transaction_date, transaction_amount, transaction_type, employment_type, gender, age_group, loan_default
    """
    np.random.seed(42)
    random.seed(42)
    
    # 12 months spanning backwards from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    
    # Customer Profiles
    # 0: Stable Salaried (Safe)
    # 1: High-spending Salaried (Medium Risk)
    # 2: Stable Gig/Freelancer (Medium Risk)
    # 3: Volatile Gig (High Risk)
    # 4: Low-Income Worker (High Risk)
    
    profiles = np.random.choice([0, 1, 2, 3, 4], size=num_customers, p=[0.25, 0.15, 0.20, 0.20, 0.20])
    
    genders = ['Male', 'Female', 'Non-Binary']
    age_groups = ['18-25', '26-40', '41-60', '60+']
    
    transactions = []
    customer_metadata = []
    
    for i in range(num_customers):
        customer_id = f"CUST_{i:04d}"
        prof = profiles[i]
        
        gender = np.random.choice(genders, p=[0.55, 0.40, 0.05])
        age_group = np.random.choice(age_groups, p=[0.2, 0.5, 0.25, 0.05])
        
        if prof == 0:
            emp_type = 'salaried'
            monthly_income_mean = np.random.uniform(50000, 120000)
            income_volatility = np.random.uniform(0.01, 0.05)
            expense_ratio_mean = np.random.uniform(0.4, 0.7) # High savings
            expense_volatility = np.random.uniform(0.1, 0.2)
            monthly_tx_count = np.random.randint(15, 30)
            default_prob = 0.02
        elif prof == 1:
            emp_type = 'salaried'
            monthly_income_mean = np.random.uniform(60000, 150000)
            income_volatility = np.random.uniform(0.02, 0.08)
            expense_ratio_mean = np.random.uniform(0.8, 1.05) # High spending, low savings
            expense_volatility = np.random.uniform(0.3, 0.6)
            monthly_tx_count = np.random.randint(30, 60)
            default_prob = 0.25
        elif prof == 2:
            emp_type = 'gig'
            monthly_income_mean = np.random.uniform(30000, 70000)
            income_volatility = np.random.uniform(0.15, 0.3)
            expense_ratio_mean = np.random.uniform(0.5, 0.75) # Good savings discipline
            expense_volatility = np.random.uniform(0.2, 0.4)
            monthly_tx_count = np.random.randint(20, 40)
            default_prob = 0.10
        elif prof == 3:
            emp_type = 'gig'
            monthly_income_mean = np.random.uniform(20000, 50000)
            income_volatility = np.random.uniform(0.4, 0.8) # Highly volatile income
            expense_ratio_mean = np.random.uniform(0.9, 1.2) # Spending exceeds income often
            expense_volatility = np.random.uniform(0.5, 0.9)
            monthly_tx_count = np.random.randint(25, 50)
            default_prob = 0.45
        else:
            emp_type = 'self-employed'
            monthly_income_mean = np.random.uniform(15000, 35000)
            income_volatility = np.random.uniform(0.1, 0.25)
            expense_ratio_mean = np.random.uniform(0.85, 1.1)
            expense_volatility = np.random.uniform(0.2, 0.5)
            monthly_tx_count = np.random.randint(10, 25)
            default_prob = 0.35
            
        is_default = np.random.choice([0, 1], p=[1-default_prob, default_prob])
        
        customer_metadata.append({
            'customer_id': customer_id,
            'employment_type': emp_type,
            'gender': gender,
            'age_group': age_group,
            'loan_default': is_default,
            'avg_monthly_income': round(monthly_income_mean, 2),
            'savings_ratio': round(1.0 - expense_ratio_mean, 2)
        })
        
        # Generate transactions for each month
        for month in range(months):
            # Dynamic month income
            current_income = max(1000, np.random.normal(monthly_income_mean, monthly_income_mean * income_volatility))
            month_start = start_date + timedelta(days=30 * month)
            
            # Income transactions
            income_tx_count = 1 if emp_type == 'salaried' else np.random.randint(2, 6)
            income_amounts = np.random.dirichlet(np.ones(income_tx_count)) * current_income
            
            for amt in income_amounts:
                tx_date = month_start + timedelta(days=np.random.randint(0, 28))
                transactions.append([
                    customer_id, tx_date.strftime("%Y-%m-%d"), round(amt, 2), 'income'
                ])
                
            # Expense transactions
            current_expense = max(500, np.random.normal(current_income * expense_ratio_mean, current_income * expense_volatility))
            
            expense_tx_count = max(1, monthly_tx_count - income_tx_count)
            expense_amounts = np.random.dirichlet(np.ones(expense_tx_count)) * current_expense
            
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
    
    # Ensure no nulls
    df_final = df_final.fillna(0)
    
    return df_final

if __name__ == "__main__":
    print("Generating 500 customer synthetic dataset (12 months)...")
    df = generate_synthetic_data(500, 12)
    print(f"Dataset generated. Shape: {df.shape}")
    print(f"Total Transactions: {len(df)}")
    
    # Save CSV
    output_path = 'realistic_fintech_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
