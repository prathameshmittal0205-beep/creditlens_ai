import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_customers=1000, months=12):
    """
    Generates synthetic transaction-level data for CreditLens AI.
    Features: customer_id, transaction_date, transaction_amount, transaction_type, employment_type, loan_default
    """
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=30 * months)
    
    customers = []
    
    # 3 types of employment: gig, salaried, self-employed
    employment_types = ['gig'] * int(num_customers * 0.5) + ['salaried'] * int(num_customers * 0.3) + ['self-employed'] * int(num_customers * 0.2)
    np.random.shuffle(employment_types)
    
    transactions = []
    
    for i in range(num_customers):
        customer_id = f"CUST_{i:04d}"
        emp_type = employment_types[i]
        
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
                    customer_id, tx_date.strftime("%Y-%m-%d"), round(amt, 2), 'income', emp_type, is_default
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
                    customer_id, tx_date.strftime("%Y-%m-%d"), round(amt, 2), 'expense', emp_type, is_default
                ])

    df = pd.DataFrame(transactions, columns=['customer_id', 'transaction_date', 'transaction_amount', 'transaction_type', 'employment_type', 'loan_default'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values(by=['customer_id', 'transaction_date']).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Generated {len(df)} transactions for {df['customer_id'].nunique()} customers.")
    print(df.head())
