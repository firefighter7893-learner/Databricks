# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Project Overview
# MAGIC %md
# MAGIC # MBA Analytics & Operations Project - Prescriptive Recommendations
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC This notebook translates predictive insights into actionable business recommendations across inventory management, customer retention, product portfolio, and sales optimization.
# MAGIC
# MAGIC ### Recommendation Areas:
# MAGIC
# MAGIC 1. **Inventory Optimization**
# MAGIC    - Reorder recommendations based on demand forecasts
# MAGIC    - Stock level optimization by ABC-VED classification
# MAGIC    - Working capital allocation strategies
# MAGIC
# MAGIC 2. **Customer Retention & Growth**
# MAGIC    - Targeted campaigns for high-value at-risk customers
# MAGIC    - Personalized engagement by CLV tier
# MAGIC    - Win-back strategies for churned customers
# MAGIC
# MAGIC 3. **Product Portfolio Optimization**
# MAGIC    - Phase-out recommendations for high-return products
# MAGIC    - Promotion strategies for growing products
# MAGIC    - Pricing and margin optimization
# MAGIC
# MAGIC 4. **Sales Force Optimization**
# MAGIC    - Territory and account allocation
# MAGIC    - Focus areas and performance targets
# MAGIC    - Compensation alignment with customer value
# MAGIC
# MAGIC 5. **Operational Efficiency**
# MAGIC    - Process improvements based on data insights
# MAGIC    - Cost reduction opportunities
# MAGIC    - Service level optimization
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Import Libraries
# 📦 SETUP: Import Required Libraries
# What: Load Python libraries for data analysis and recommendations
# Why: pandas for data manipulation, numpy for calculations, datetime for business logic
# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully")
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")

# COMMAND ----------

# DBTITLE 1,Section 1 Header
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 1. Inventory Optimization Recommendations
# MAGIC
# MAGIC ### 1.1 Load Predictive Data

# COMMAND ----------

# DBTITLE 1,Load Inventory Forecast Data
# MAGIC %sql
# MAGIC -- 📊 INVENTORY DATA PREPARATION
# MAGIC -- What: Calculate average daily demand and variability for each product from historical sales
# MAGIC -- Input: commercial.base.products, sales, categories tables
# MAGIC -- Output: product_demand_summary temp view with demand statistics
# MAGIC -- Formula: avg_daily_demand = AVG(Quantity), demand_std_dev = STDDEV(Quantity)
# MAGIC -- Business Use: Foundation for calculating Safety Stock, ROP, and EOQ in next cells
# MAGIC     
# MAGIC -- Recreate inventory forecast from Predictive Models notebook
# MAGIC -- Load product demand forecasts
# MAGIC CREATE OR REPLACE TEMP VIEW product_demand_summary AS
# MAGIC SELECT 
# MAGIC   p.ProductID,
# MAGIC   p.ProductName,
# MAGIC   p.CategoryID,
# MAGIC   c.CategoryName,
# MAGIC   p.Price,
# MAGIC   p.VitalityDays,
# MAGIC   -- Calculate average daily demand from historical data
# MAGIC   AVG(s.Quantity) as avg_daily_demand,
# MAGIC   STDDEV(s.Quantity) as demand_std_dev,
# MAGIC   COUNT(DISTINCT DATE(s.SalesDate)) as days_with_sales,
# MAGIC   SUM(s.Quantity) as total_quantity_sold
# MAGIC FROM commercial.base.products p
# MAGIC LEFT JOIN commercial.base.categories c ON p.CategoryID = c.CategoryID
# MAGIC LEFT JOIN commercial.base.sales s ON p.ProductID = s.ProductID
# MAGIC GROUP BY p.ProductID, p.ProductName, p.CategoryID, c.CategoryName, p.Price, p.VitalityDays
# MAGIC HAVING COUNT(DISTINCT DATE(s.SalesDate)) >= 30;  -- Products with at least 30 days of sales
# MAGIC
# MAGIC SELECT COUNT(*) as total_products FROM product_demand_summary;

# COMMAND ----------

# DBTITLE 1,Calculate Reorder Recommendations
# 📐 INVENTORY CALCULATIONS: Safety Stock, ROP, EOQ
# What: Calculate optimal inventory levels for each product using standard operations formulas
# Inputs: avg_daily_demand, demand_std_dev, lead_time (7 days), service_level (95%)
# Formulas:
#   • Safety Stock = Z-score × √(Lead Time) × Std Dev  [Z=1.65 for 95% service level]
#   • Reorder Point (ROP) = (Avg Daily Demand × Lead Time) + Safety Stock
#   • Order Quantity = Avg Daily Demand × Review Period (14 days)
# Business Use: Tells when to reorder (ROP) and how much to order (Order Quantity)
# Calculate inventory requirements and reorder recommendations

# Load product demand data
demand_data = spark.sql("""
  SELECT 
    ProductID,
    ProductName,
    CategoryName,
    Price,
    VitalityDays,
    avg_daily_demand,
    demand_std_dev,
    total_quantity_sold
  FROM product_demand_summary
""").toPandas()

print(f"Analyzing {len(demand_data):,} products for inventory recommendations")

# Inventory parameters
lead_time_days = 7  # Supplier lead time
service_level_z = 1.65  # 95% service level
review_period_days = 14  # How often we review inventory

# Calculate safety stock and reorder points
demand_data['safety_stock'] = (
    service_level_z * 
    np.sqrt(lead_time_days) * 
    demand_data['demand_std_dev']
).fillna(0).round(0)

demand_data['reorder_point'] = (
    (demand_data['avg_daily_demand'] * lead_time_days) + 
    demand_data['safety_stock']
).round(0)

# Calculate economic order quantity (simplified)
demand_data['order_quantity'] = (
    demand_data['avg_daily_demand'] * review_period_days
).round(0)

# For perishable items, cap at shelf life
demand_data['max_inventory'] = np.where(
    demand_data['VitalityDays'] > 0,
    np.minimum(
        demand_data['reorder_point'] + demand_data['order_quantity'],
        demand_data['avg_daily_demand'] * demand_data['VitalityDays'] * 0.8  # 80% of shelf life
    ),
    demand_data['reorder_point'] + demand_data['order_quantity']
).round(0)

# Calculate inventory value
demand_data['reorder_point_value'] = demand_data['reorder_point'] * demand_data['Price']
demand_data['order_quantity_value'] = demand_data['order_quantity'] * demand_data['Price']

print(f"\nTotal Reorder Point Investment: ${demand_data['reorder_point_value'].sum():,.2f}")
print(f"Average Safety Stock: {demand_data['safety_stock'].mean():.0f} units")
print(f"Average Reorder Point: {demand_data['reorder_point'].mean():.0f} units")

# COMMAND ----------

# DBTITLE 1,ABC-VED Classification for Recommendations
# 🏷️ MULTI-DIMENSIONAL CLASSIFICATION: ABC-VED-FSN Analysis
# What: Classify products by 3 dimensions to determine inventory policy
# Classifications:
#   • ABC: Based on total value (A=high revenue, B=medium, C=low)
#   • VED: Based on criticality (V=vital/perishable, E=essential/high demand, D=desirable)
#   • FSN: Based on velocity (Fast/Slow/Non-moving)
# Output: Inventory policy recommendation per product (e.g., "JIT delivery", "weekly monitoring")
# Business Use: Determines how frequently to monitor, order quantity size, and control level
# Classify products by value (ABC) and criticality (VED approximation)

# ABC Classification (by value)
demand_data['inventory_value'] = demand_data['total_quantity_sold'] * demand_data['Price']
demand_data['abc_class'] = pd.qcut(
    demand_data['inventory_value'], 
    q=3, 
    labels=['C', 'B', 'A'],
    duplicates='drop'
)

# VED Classification (based on shelf life and demand velocity)
# Calculate 75th percentile threshold first
demand_threshold_75 = demand_data['avg_daily_demand'].quantile(0.75)

def classify_ved(row):
    if row['VitalityDays'] > 0 and row['VitalityDays'] < 30:
        return 'V'  # Vital - perishable items
    elif row['avg_daily_demand'] > demand_threshold_75:
        return 'E'  # Essential - high demand
    else:
        return 'D'  # Desirable - normal items

demand_data['ved_class'] = demand_data.apply(classify_ved, axis=1)

# FSN Classification (Fast/Slow/Non-moving based on velocity)
demand_data['fsn_class'] = pd.qcut(
    demand_data['avg_daily_demand'], 
    q=3, 
    labels=['Non-Moving', 'Slow-Moving', 'Fast-Moving'],
    duplicates='drop'
)

# Generate inventory policy recommendations
def recommend_inventory_policy(row):
    abc = row['abc_class']
    ved = row['ved_class']
    fsn = row['fsn_class']
    
    if abc == 'A' and ved == 'V':
        return 'CRITICAL: JIT delivery, daily monitoring, small frequent orders'
    elif abc == 'A' and fsn == 'Fast-Moving':
        return 'HIGH PRIORITY: Automated reorder, weekly monitoring, bulk discounts'
    elif abc == 'A':
        return 'HIGH VALUE: Standard reorder process, bi-weekly review'
    elif ved == 'V':
        return 'PERISHABLE: Frequent small orders, tight expiry tracking'
    elif fsn == 'Non-Moving':
        return 'LOW PRIORITY: Monthly review, consider discontinuation'
    elif abc == 'C' and fsn == 'Slow-Moving':
        return 'OPTIMIZE: Reduce stock levels, consider make-to-order'
    else:
        return 'STANDARD: Normal reorder process, monthly review'

demand_data['inventory_policy'] = demand_data.apply(recommend_inventory_policy, axis=1)

print("\n=== Inventory Policy Distribution ===")
print(demand_data['inventory_policy'].value_counts())

print("\n=== ABC-VED Matrix ===")
abc_ved_matrix = pd.crosstab(demand_data['abc_class'], demand_data['ved_class'])
print(abc_ved_matrix)

# COMMAND ----------

# DBTITLE 1,Priority Reorder List
# ⚠️ ACTION REQUIRED: Priority Reorder List
# What: Identify products currently below ROP that need immediate reordering
# Logic: Simulate current stock at 50% of ROP, then flag items needing reorder
# Priority Score: Weighted combination of value (40%), velocity (30%), investment (30%)
# Output: Ranked list of products to reorder immediately, sorted by business impact
# Business Use: Daily/weekly reorder checklist for procurement team
# Generate priority reorder recommendations

# Assume current stock is at 50% of reorder point (simulation)
demand_data['current_stock_simulated'] = (demand_data['reorder_point'] * 0.5).round(0)
demand_data['needs_reorder'] = demand_data['current_stock_simulated'] <= demand_data['reorder_point']

# Calculate priority score
demand_data['reorder_priority'] = (
    (demand_data['inventory_value'] / demand_data['inventory_value'].max()) * 0.4 +  # 40% value weight
    (demand_data['avg_daily_demand'] / demand_data['avg_daily_demand'].max()) * 0.3 +  # 30% velocity weight
    (demand_data['reorder_point_value'] / demand_data['reorder_point_value'].max()) * 0.3  # 30% investment weight
)

# Products needing reorder
reorder_needed = demand_data[demand_data['needs_reorder']].copy()
reorder_needed = reorder_needed.sort_values('reorder_priority', ascending=False)

print(f"\n{'='*80}")
print("IMMEDIATE REORDER RECOMMENDATIONS")
print(f"{'='*80}")
print(f"\nProducts requiring reorder: {len(reorder_needed):,}")
print(f"Total reorder investment: ${reorder_needed['order_quantity_value'].sum():,.2f}")

print("\n=== Top 20 Priority Reorders ===")
top_20_reorders = reorder_needed.head(20)[[
    'ProductID', 'ProductName', 'abc_class', 'ved_class',
    'current_stock_simulated', 'reorder_point', 'order_quantity', 'order_quantity_value'
]].round(2)

for idx, row in top_20_reorders.iterrows():
    print(f"\n{row['ProductName'][:40]:40s} (ID: {row['ProductID']})")
    print(f"  Class: {row['abc_class']}-{row['ved_class']} | Current: {row['current_stock_simulated']:.0f} | ROP: {row['reorder_point']:.0f} | Order: {row['order_quantity']:.0f} units (${row['order_quantity_value']:,.2f})")

# COMMAND ----------

# DBTITLE 1,Inventory Investment Optimization
# 💰 WORKING CAPITAL OPTIMIZATION: Investment by ABC Class
# What: Analyze total inventory investment by ABC classification
# Calculates: ROP investment, Order investment, Annual value by class
# Recommendations:
#   • A-class: Tight control, automated systems, VMI (Vendor Managed Inventory)
#   • B-class: Standard processes, periodic review
#   • C-class: Simplify ordering, larger/less frequent orders, consider discontinuation
# Savings Opportunities: Reduce C-class safety stock, eliminate non-movers, JIT for perishables
# Business Use: Strategic decisions on where to allocate working capital
# Working capital optimization recommendations

print(f"\n{'='*80}")
print("INVENTORY INVESTMENT OPTIMIZATION")
print(f"{'='*80}")

# Analyze by ABC class
abc_investment = demand_data.groupby('abc_class').agg({
    'ProductID': 'count',
    'reorder_point_value': 'sum',
    'order_quantity_value': 'sum',
    'inventory_value': 'sum'
}).round(2)

abc_investment.columns = ['Product_Count', 'ROP_Investment', 'Order_Investment', 'Annual_Value']
abc_investment['Percent_of_Value'] = (abc_investment['Annual_Value'] / abc_investment['Annual_Value'].sum() * 100).round(1)

print("\n=== Investment by ABC Class ===")
print(abc_investment)

print("\n=== Recommendations by Class ===")
print("\nA-Class Items (High Value):")
print("  • Tight inventory control with automated systems")
print("  • Negotiate bulk discounts with suppliers")
print("  • Consider VMI (Vendor Managed Inventory) for top items")
print(f"  • Investment: ${abc_investment.loc['A', 'ROP_Investment']:,.2f}")

print("\nB-Class Items (Medium Value):")
print("  • Standard reorder processes with periodic review")
print("  • Balance between service level and holding costs")
print("  • Quarterly supplier negotiations")
print(f"  • Investment: ${abc_investment.loc['B', 'ROP_Investment']:,.2f}")

print("\nC-Class Items (Low Value):")
print("  • Simplify ordering process - larger, less frequent orders")
print("  • Consider consignment inventory")
print("  • Evaluate discontinuation of slowest movers")
print(f"  • Investment: ${abc_investment.loc['C', 'ROP_Investment']:,.2f}")

# Optimization opportunities
total_investment = demand_data['reorder_point_value'].sum()
print(f"\n=== Working Capital Optimization Opportunities ===")
print(f"Current Total Investment: ${total_investment:,.2f}")
print(f"\nPotential Savings:")
print(f"  • Reduce C-class safety stock by 20%: ${abc_investment.loc['C', 'ROP_Investment'] * 0.20:,.2f}")
print(f"  • Eliminate non-moving items: ${demand_data[demand_data['fsn_class'] == 'Non-Moving']['reorder_point_value'].sum() * 0.5:,.2f}")
print(f"  • JIT for perishables (reduce 30%): ${demand_data[demand_data['VitalityDays'] < 30]['reorder_point_value'].sum() * 0.30:,.2f}")

# COMMAND ----------

# DBTITLE 1,Customer Section
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 2. Customer Retention & Growth Strategies
# MAGIC
# MAGIC ### 2.1 Load Customer Predictions

# COMMAND ----------

# DBTITLE 1,Load Customer Metrics
# MAGIC %sql
# MAGIC -- 👥 CUSTOMER SEGMENTATION DATA PREPARATION
# MAGIC -- What: Calculate customer value metrics and risk levels from transaction history
# MAGIC -- Metrics Calculated:
# MAGIC --   • Total transactions, revenue, AOV (Average Order Value)
# MAGIC --   • Recency (days since last purchase)
# MAGIC --   • CLV Tier (5 tiers: Very High to Very Low based on revenue percentiles)
# MAGIC --   • Churn Risk (Active, Low/Medium/High Risk based on recency)
# MAGIC -- Output: customer_analysis temp view for retention strategy development
# MAGIC -- Business Use: Identify high-value at-risk customers for targeted intervention
# MAGIC     
# MAGIC -- Recreate customer metrics for recommendations
# MAGIC CREATE OR REPLACE TEMP VIEW customer_analysis AS
# MAGIC SELECT 
# MAGIC   CustomerID,
# MAGIC   COUNT(DISTINCT SalesID) as total_transactions,
# MAGIC   SUM(TotalPrice) as total_revenue,
# MAGIC   AVG(TotalPrice) as avg_order_value,
# MAGIC   MAX(SalesDate) as last_purchase_date,
# MAGIC   DATEDIFF(CURRENT_DATE(), MAX(SalesDate)) as days_since_last_purchase,
# MAGIC   COUNT(DISTINCT ProductID) as unique_products_purchased,
# MAGIC   DATEDIFF(MAX(SalesDate), MIN(SalesDate)) as customer_age_days,
# MAGIC   -- CLV Tier (simplified based on revenue)
# MAGIC   CASE 
# MAGIC     WHEN SUM(TotalPrice) >= PERCENTILE(SUM(TotalPrice), 0.8) OVER() THEN 'Very High'
# MAGIC     WHEN SUM(TotalPrice) >= PERCENTILE(SUM(TotalPrice), 0.6) OVER() THEN 'High'
# MAGIC     WHEN SUM(TotalPrice) >= PERCENTILE(SUM(TotalPrice), 0.4) OVER() THEN 'Medium'
# MAGIC     WHEN SUM(TotalPrice) >= PERCENTILE(SUM(TotalPrice), 0.2) OVER() THEN 'Low'
# MAGIC     ELSE 'Very Low'
# MAGIC   END as clv_tier,
# MAGIC   -- Churn Risk
# MAGIC   CASE 
# MAGIC     WHEN DATEDIFF(CURRENT_DATE(), MAX(SalesDate)) > 90 THEN 'High Risk'
# MAGIC     WHEN DATEDIFF(CURRENT_DATE(), MAX(SalesDate)) > 60 THEN 'Medium Risk'
# MAGIC     WHEN DATEDIFF(CURRENT_DATE(), MAX(SalesDate)) > 30 THEN 'Low Risk'
# MAGIC     ELSE 'Active'
# MAGIC   END as churn_risk
# MAGIC FROM commercial.base.sales
# MAGIC GROUP BY CustomerID
# MAGIC HAVING COUNT(DISTINCT SalesID) >= 2;
# MAGIC
# MAGIC SELECT COUNT(*) as total_customers, 
# MAGIC        COUNT(CASE WHEN churn_risk IN ('High Risk', 'Medium Risk') THEN 1 END) as at_risk_customers
# MAGIC FROM customer_analysis;

# COMMAND ----------

# DBTITLE 1,Customer Segmentation Strategy
# 🎯 CUSTOMER SEGMENTATION MATRIX: CLV × Churn Risk
# What: Create 2D segmentation matrix combining Customer Lifetime Value with Churn Risk
# Matrix Dimensions:
#   • X-axis: CLV Tier (Very High, High, Medium, Low, Very Low)
#   • Y-axis: Churn Risk (Active, Low Risk, Medium Risk, High Risk)
# Priority Segment: High/Very High CLV + High/Medium Risk = Immediate action required
# Output: Count and revenue at risk for each segment
# Business Use: Prioritize retention efforts on highest-value at-risk customers
# Load customer data and develop retention strategies

customer_data = spark.sql("""
  SELECT 
    CustomerID,
    total_transactions,
    total_revenue,
    avg_order_value,
    days_since_last_purchase,
    clv_tier,
    churn_risk,
    unique_products_purchased
  FROM customer_analysis
""").toPandas()

print(f"Analyzing {len(customer_data):,} customers for retention strategies\n")

# Segment customers by CLV and Churn Risk
segmentation = pd.crosstab(
    customer_data['clv_tier'], 
    customer_data['churn_risk'],
    margins=True
)

print("=== Customer Segmentation Matrix ===")
print(segmentation)
print()

# High-value at-risk customers (priority for retention)
high_value_risk = customer_data[
    (customer_data['clv_tier'].isin(['High', 'Very High'])) &
    (customer_data['churn_risk'].isin(['High Risk', 'Medium Risk']))
].copy()

print(f"\n{'='*80}")
print("HIGH PRIORITY: High-Value At-Risk Customers")
print(f"{'='*80}")
print(f"Count: {len(high_value_risk):,} customers")
print(f"Revenue at Risk: ${high_value_risk['total_revenue'].sum():,.2f}")
print(f"Average Revenue per Customer: ${high_value_risk['total_revenue'].mean():,.2f}")

# COMMAND ----------

# DBTITLE 1,Retention Campaign Recommendations
# 📧 RETENTION CAMPAIGNS: 4 Targeted Strategies by Segment
# What: Design specific retention campaigns for each customer segment with ROI estimates
# Campaigns:
#   1. Win-Back High-Value: Personal outreach + 15-20% discount for at-risk high CLV
#   2. VIP Loyalty Program: Exclusive benefits for active high-value customers
#   3. Growth Campaign: Cross-sell and upsell for medium-value active customers
#   4. Mass Re-Activation: Low-cost automated email for inactive low-value customers
# Each campaign includes: Target size, cost, expected response rate, revenue potential
# Business Use: Marketing team action plan with clear budgets and expected returns
# Develop targeted retention campaigns

print(f"\n{'='*80}")
print("CUSTOMER RETENTION CAMPAIGN RECOMMENDATIONS")
print(f"{'='*80}\n")

# Campaign 1: High-Value At-Risk
print("\n📧 CAMPAIGN 1: Win-Back High-Value Customers")
print("-" * 80)
print(f"Target Segment: High/Very High CLV + High/Medium Churn Risk")
print(f"Target Size: {len(high_value_risk):,} customers")
print(f"Revenue Potential: ${high_value_risk['total_revenue'].sum():,.2f}")
print("\nRecommended Actions:")
print("  1. Personalized outreach from account manager")
print("  2. Exclusive 15-20% discount on next purchase")
print("  3. VIP service recovery package")
print("  4. Direct call within 48 hours")
print("  5. Survey to understand concerns")
print(f"\nEstimated Cost: ${len(high_value_risk) * 50:,.2f} (outreach + discount)")
print(f"Expected Recovery Rate: 30-40%")
print(f"Expected Revenue Recovery: ${high_value_risk['total_revenue'].sum() * 0.35:,.2f}")

# Campaign 2: Active High-Value (Retention)
active_high_value = customer_data[
    (customer_data['clv_tier'].isin(['High', 'Very High'])) &
    (customer_data['churn_risk'] == 'Active')
]

print("\n\n🏆 CAMPAIGN 2: Loyalty Program for Active High-Value")
print("-" * 80)
print(f"Target Segment: High/Very High CLV + Active")
print(f"Target Size: {len(active_high_value):,} customers")
print(f"Annual Revenue: ${active_high_value['total_revenue'].sum():,.2f}")
print("\nRecommended Actions:")
print("  1. Launch VIP loyalty tier with exclusive benefits")
print("  2. Early access to new products")
print("  3. Dedicated customer service line")
print("  4. Quarterly appreciation gifts")
print("  5. Referral rewards (bonus for bringing new customers)")
print(f"\nProgram Cost: ${len(active_high_value) * 30:,.2f} annually")
print(f"Expected Retention Lift: 15%")
print(f"Expected Additional Revenue: ${active_high_value['total_revenue'].sum() * 0.15:,.2f}")

# Campaign 3: Medium-Value Growth
medium_active = customer_data[
    (customer_data['clv_tier'] == 'Medium') &
    (customer_data['churn_risk'] == 'Active')
]

print("\n\n📈 CAMPAIGN 3: Growth Campaign for Medium-Value Customers")
print("-" * 80)
print(f"Target Segment: Medium CLV + Active")
print(f"Target Size: {len(medium_active):,} customers")
print(f"Annual Revenue: ${medium_active['total_revenue'].sum():,.2f}")
print("\nRecommended Actions:")
print("  1. Cross-sell complementary products (10% discount bundle)")
print("  2. Increase order frequency with subscription incentives")
print("  3. Educational content about product benefits")
print("  4. Seasonal promotion campaigns")
print("  5. Gamified rewards program")
print(f"\nCampaign Cost: ${len(medium_active) * 10:,.2f}")
print(f"Expected Upgrade to High CLV: 20%")
print(f"Potential Revenue Increase: ${medium_active['total_revenue'].sum() * 0.20:,.2f}")

# Campaign 4: Re-activation of Inactive Low-Value
inactive_low = customer_data[
    (customer_data['clv_tier'].isin(['Low', 'Very Low'])) &
    (customer_data['churn_risk'] == 'High Risk')
]

print("\n\n🔄 CAMPAIGN 4: Mass Re-Activation (Low-Cost)")
print("-" * 80)
print(f"Target Segment: Low/Very Low CLV + High Risk")
print(f"Target Size: {len(inactive_low):,} customers")
print("\nRecommended Actions:")
print("  1. Automated email campaign with 10% discount code")
print("  2. Highlight new products or bestsellers")
print("  3. Free shipping offer on next order")
print("  4. Simple survey with incentive")
print(f"\nCampaign Cost: ${len(inactive_low) * 2:,.2f} (low-cost automation)")
print(f"Expected Response Rate: 5-8%")
print(f"Break-even if {len(inactive_low) * 0.05:.0f} customers make one purchase")

# COMMAND ----------

# DBTITLE 1,Customer Lifetime Value Optimization
# 📈 CLV GROWTH STRATEGIES: Tailored by Customer Tier
# What: Develop tier-specific strategies to increase customer lifetime value
# Strategies by Tier:
#   • Low CLV: Increase purchase frequency (+1 purchase/year) via reminders, subscriptions
#   • Medium CLV: Increase basket size (+20% AOV) via bundles, cross-sell, upsell
#   • High CLV: Maximize retention (95%+) and referrals via VIP programs, exclusive events
# Calculates: Current revenue, potential increase, tactics for each tier
# Output: Total CLV optimization opportunity with investment and ROI estimates
# Business Use: Strategic roadmap for customer value growth over 12-24 months
# CLV optimization strategies

print(f"\n{'='*80}")
print("CUSTOMER LIFETIME VALUE OPTIMIZATION")
print(f"{'='*80}\n")

# Analyze purchase patterns
avg_by_tier = customer_data.groupby('clv_tier').agg({
    'CustomerID': 'count',
    'total_revenue': ['sum', 'mean'],
    'avg_order_value': 'mean',
    'total_transactions': 'mean',
    'unique_products_purchased': 'mean'
}).round(2)

print("=== Customer Metrics by CLV Tier ===")
print(avg_by_tier)

# Calculate potential CLV increase
print("\n\n=== CLV Growth Opportunities ===")
print("-" * 80)

for tier in ['Low', 'Medium', 'High']:
    if tier not in customer_data['clv_tier'].values:
        continue
        
    tier_data = customer_data[customer_data['clv_tier'] == tier]
    tier_revenue = tier_data['total_revenue'].sum()
    tier_count = len(tier_data)
    
    print(f"\n{tier} CLV Tier:")
    print(f"  Current Annual Revenue: ${tier_revenue:,.2f}")
    print(f"  Customer Count: {tier_count:,}")
    
    # Strategy recommendations
    if tier == 'Low':
        print("  Strategy: Increase purchase frequency")
        print(f"    • Target: +1 purchase per customer per year")
        print(f"    • Potential Revenue: ${tier_data['avg_order_value'].mean() * tier_count:,.2f}")
        print("    • Tactics: Automated reminders, subscription offers, loyalty points")
    elif tier == 'Medium':
        print("  Strategy: Increase basket size and product range")
        print(f"    • Target: +20% average order value")
        print(f"    • Potential Revenue: ${tier_revenue * 0.20:,.2f}")
        print("    • Tactics: Bundle offers, cross-sell, upsell premium products")
    elif tier == 'High':
        print("  Strategy: Maximize retention and referrals")
        print(f"    • Target: 95% retention + 0.5 referrals per customer")
        print(f"    • Retention Value: ${tier_revenue * 0.95:,.2f}")
        print(f"    • Referral Potential: ${tier_data['total_revenue'].mean() * 0.5 * tier_count:,.2f}")
        print("    • Tactics: VIP program, exclusive events, referral bonuses")

# Total opportunity
total_current = customer_data['total_revenue'].sum()
print(f"\n\n{'='*80}")
print(f"TOTAL CLV OPTIMIZATION OPPORTUNITY")
print(f"{'='*80}")
print(f"Current Annual Revenue: ${total_current:,.2f}")
print(f"Potential Revenue Increase (10-15% target): ${total_current * 0.125:,.2f}")
print(f"Investment Required: ${len(customer_data) * 15:,.2f} (marketing + programs)")
print(f"Expected ROI: {(total_current * 0.125) / (len(customer_data) * 15):.1f}x")

# COMMAND ----------

# DBTITLE 1,Product Section
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 3. Product Portfolio Optimization
# MAGIC
# MAGIC ### 3.1 Product Performance Analysis

# COMMAND ----------

# DBTITLE 1,Product Performance Metrics
# MAGIC %sql
# MAGIC -- 📦 PRODUCT PORTFOLIO ANALYSIS
# MAGIC -- What: Calculate comprehensive performance metrics for each product
# MAGIC -- Metrics:
# MAGIC --   • Revenue: Total sales, adjusted for returns (adjusted_revenue)
# MAGIC --   • Return Rate: Percentage of units returned (quality indicator)
# MAGIC --   • Customer Reach: Unique customers per product
# MAGIC --   • Discount Dependency: Average discount percentage
# MAGIC -- Product Classification: STAR, STRONG, STANDARD, PROBLEM, DOG, DISCONTINUE
# MAGIC -- Output: product_performance temp view for portfolio optimization decisions
# MAGIC -- Business Use: Identify products to grow, maintain, fix, or discontinue
# MAGIC     
# MAGIC -- Analyze product performance for optimization recommendations
# MAGIC CREATE OR REPLACE TEMP VIEW product_performance AS
# MAGIC WITH sales_metrics AS (
# MAGIC   SELECT 
# MAGIC     p.ProductID,
# MAGIC     p.ProductName,
# MAGIC     p.Price,
# MAGIC     c.CategoryName,
# MAGIC     p.VitalityDays,
# MAGIC     COUNT(*) as transaction_count,
# MAGIC     SUM(s.Quantity) as total_quantity_sold,
# MAGIC     SUM(s.TotalPrice) as total_revenue,
# MAGIC     AVG(s.Discount) as avg_discount,
# MAGIC     COUNT(DISTINCT s.CustomerID) as unique_customers
# MAGIC   FROM commercial.base.products p
# MAGIC   LEFT JOIN commercial.base.sales s ON p.ProductID = s.ProductID
# MAGIC   LEFT JOIN commercial.base.categories c ON p.CategoryID = c.CategoryID
# MAGIC   GROUP BY p.ProductID, p.ProductName, p.Price, c.CategoryName, p.VitalityDays
# MAGIC ),
# MAGIC return_metrics AS (
# MAGIC   SELECT 
# MAGIC     ProductID,
# MAGIC     COUNT(*) as return_count,
# MAGIC     SUM(Quantity) as returned_quantity
# MAGIC   FROM commercial.base.sales_cancelled_returned
# MAGIC   GROUP BY ProductID
# MAGIC )
# MAGIC SELECT 
# MAGIC   sm.*,
# MAGIC   COALESCE(rm.return_count, 0) as return_count,
# MAGIC   COALESCE(rm.returned_quantity, 0) as returned_quantity,
# MAGIC   COALESCE(rm.return_count * 100.0 / NULLIF(sm.transaction_count, 0), 0) as return_rate_pct,
# MAGIC   -- Profitability proxy (revenue - discount impact)
# MAGIC   total_revenue * (1 - avg_discount) as adjusted_revenue,
# MAGIC   -- Growth indicator (comparing recent vs older sales would require date-based logic)
# MAGIC   total_revenue / NULLIF(transaction_count, 0) as revenue_per_transaction
# MAGIC FROM sales_metrics sm
# MAGIC LEFT JOIN return_metrics rm ON sm.ProductID = rm.ProductID
# MAGIC WHERE sm.transaction_count > 0;
# MAGIC
# MAGIC SELECT COUNT(*) as total_products, 
# MAGIC        SUM(total_revenue) as total_revenue,
# MAGIC        AVG(return_rate_pct) as avg_return_rate
# MAGIC FROM product_performance;

# COMMAND ----------

# DBTITLE 1,Product Optimization Recommendations
# 🏷️ PRODUCT CLASSIFICATION: Portfolio Strategy Matrix
# What: Classify each product into strategic categories using revenue and return rate
# Classifications:
#   • STAR: High revenue + low returns (invest and grow)
#   • STRONG: Good revenue + low returns (maintain)
#   • STANDARD: Average performance (monitor)
#   • PROBLEM: High revenue + high returns (fix quality issues)
#   • DOG: Low revenue + high returns (phase out)
#   • DISCONTINUE: Very low revenue (eliminate)
# Output: Revenue distribution and product count by class
# Business Use: Strategic portfolio decisions - where to invest, maintain, or cut
# Analyze product performance and generate recommendations

product_perf = spark.sql("""
  SELECT 
    ProductID,
    ProductName,
    Price,
    CategoryName,
    total_quantity_sold,
    total_revenue,
    adjusted_revenue,
    avg_discount,
    return_rate_pct,
    unique_customers,
    transaction_count
  FROM product_performance
""").toPandas()

# Convert return_rate_pct to float for numerical operations
product_perf['return_rate_pct'] = pd.to_numeric(product_perf['return_rate_pct'], errors='coerce').fillna(0)

print(f"Analyzing {len(product_perf):,} products for portfolio optimization\n")

# Classify products by performance
product_perf['revenue_rank'] = product_perf['total_revenue'].rank(pct=True)
product_perf['volume_rank'] = product_perf['total_quantity_sold'].rank(pct=True)
product_perf['return_rank'] = product_perf['return_rate_pct'].rank(pct=True, ascending=False)

# Product categories
def classify_product(row):
    if row['revenue_rank'] >= 0.8 and row['return_rank'] >= 0.7:
        return 'STAR'
    elif row['revenue_rank'] >= 0.8 and row['return_rank'] < 0.3:
        return 'PROBLEM'
    elif row['revenue_rank'] < 0.2 and row['return_rank'] >= 0.7:
        return 'DOG'
    elif row['revenue_rank'] < 0.2:
        return 'DISCONTINUE'
    elif row['revenue_rank'] >= 0.6:
        return 'STRONG'
    else:
        return 'STANDARD'

product_perf['product_class'] = product_perf.apply(classify_product, axis=1)

print("=== Product Portfolio Classification ===")
print(product_perf['product_class'].value_counts())
print()

# Revenue distribution
portfolio_summary = product_perf.groupby('product_class').agg({
    'ProductID': 'count',
    'total_revenue': 'sum',
    'return_rate_pct': 'mean'
}).round(2)
portfolio_summary.columns = ['Product_Count', 'Total_Revenue', 'Avg_Return_Rate']
portfolio_summary['Revenue_Percent'] = (portfolio_summary['Total_Revenue'] / portfolio_summary['Total_Revenue'].sum() * 100).round(1)

print("\n=== Revenue by Product Class ===")
print(portfolio_summary)

# COMMAND ----------

# DBTITLE 1,Product Action Plan
# ⭐ PRODUCT PORTFOLIO ACTION PLAN: Detailed Recommendations by Category
# What: Provide specific actions for each product category with top performers/underperformers
# Actions by Category:
#   • STAR Products: Maximize investment, increase stock, aggressive marketing
#   • STRONG Products: Maintain current strategy, monitor competition
#   • PROBLEM Products: Immediate quality investigation, fix or phase out
#   • DOG/DISCONTINUE: Clearance sales, stop reordering, reallocate shelf space
# Lists top 10 products in each category with revenue/return metrics
# Business Use: Actionable product decisions for category managers and merchandising teams
# Generate specific product recommendations

print(f"\n{'='*80}")
print("PRODUCT PORTFOLIO ACTION PLAN")
print(f"{'='*80}\n")

# STAR Products - Maximize
stars = product_perf[product_perf['product_class'] == 'STAR'].nlargest(10, 'total_revenue')
print("\n⭐ STAR PRODUCTS - Maximize Investment")
print("-" * 80)
print(f"Count: {len(product_perf[product_perf['product_class'] == 'STAR'])} products")
print(f"Revenue: ${product_perf[product_perf['product_class'] == 'STAR']['total_revenue'].sum():,.2f}")
print("\nTop 10 Stars:")
for idx, row in stars.iterrows():
    print(f"  • {row['ProductName'][:50]:50s} - ${row['total_revenue']:>12,.2f}")
print("\nRecommendations:")
print("  1. Increase inventory levels (never stock out)")
print("  2. Feature prominently in marketing campaigns")
print("  3. Create product bundles with complementary items")
print("  4. Consider premium variants or line extensions")
print("  5. Expand distribution channels")

# PROBLEM Products - Fix Quality Issues
problems = product_perf[product_perf['product_class'] == 'PROBLEM']
if len(problems) > 0:
    print("\n\n⚠️ PROBLEM PRODUCTS - Address Quality/Returns")
    print("-" * 80)
    print(f"Count: {len(problems)} products")
    print(f"Revenue: ${problems['total_revenue'].sum():,.2f}")
    print(f"Average Return Rate: {problems['return_rate_pct'].mean():.1f}%")
    print("\nTop Issues:")
    for idx, row in problems.nlargest(5, 'return_rate_pct').iterrows():
        print(f"  • {row['ProductName'][:45]:45s} - Return Rate: {row['return_rate_pct']:>6.1f}%")
    print("\nRecommendations:")
    print("  1. Immediate quality audit and supplier review")
    print("  2. Analyze return reasons and customer feedback")
    print("  3. Implement quality control checks")
    print("  4. Consider product redesign or specification changes")
    print("  5. Temporary promotion pause until issues resolved")

# DISCONTINUE Candidates
discontinue = product_perf[product_perf['product_class'] == 'DISCONTINUE']
print("\n\n🗑️ DISCONTINUATION CANDIDATES")
print("-" * 80)
print(f"Count: {len(discontinue)} products")
print(f"Revenue: ${discontinue['total_revenue'].sum():,.2f} ({discontinue['total_revenue'].sum() / product_perf['total_revenue'].sum() * 100:.1f}% of total)")
print(f"Tied-up Inventory Value: ${(discontinue['total_quantity_sold'] * discontinue['Price']).sum():,.2f}")
print("\nWorst Performers:")
for idx, row in discontinue.nsmallest(10, 'total_revenue').iterrows():
    print(f"  • {row['ProductName'][:50]:50s} - ${row['total_revenue']:>10,.2f}")
print("\nRecommendations:")
print("  1. Phase out over next 6 months (clearance pricing)")
print("  2. Do not reorder - sell through existing stock")
print("  3. Reallocate shelf space to star products")
print("  4. Analyze if any are critical for specific customers")
print(f"  5. Expected working capital release: ${(discontinue['total_quantity_sold'] * discontinue['Price']).sum() * 0.3:,.2f}")

# STRONG Products - Maintain
strong = product_perf[product_perf['product_class'] == 'STRONG']
print("\n\n💪 STRONG PRODUCTS - Maintain & Grow")
print("-" * 80)
print(f"Count: {len(strong)} products")
print(f"Revenue: ${strong['total_revenue'].sum():,.2f}")
print("\nRecommendations:")
print("  1. Maintain consistent supply and quality")
print("  2. Regular promotional calendar (seasonal campaigns)")
print("  3. Cross-sell with star products")
print("  4. Monitor for any decline in performance")
print("  5. Test small price increases on select items (5-10%)")

# COMMAND ----------

# DBTITLE 1,Sales Force Section
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 4. Sales Force Optimization
# MAGIC
# MAGIC ### 4.1 Account Value Distribution

# COMMAND ----------

# DBTITLE 1,Sales Performance Analysis
# MAGIC %sql
# MAGIC -- 📄 SALES FORCE PERFORMANCE METRICS
# MAGIC -- What: Analyze each sales representative's performance across key metrics
# MAGIC -- Metrics Calculated:
# MAGIC --   • Customer portfolio size (total and high-value customers)
# MAGIC --   • Revenue generation (total and per customer)
# MAGIC --   • Transaction metrics (count, average value)
# MAGIC --   • Order size (total units sold)
# MAGIC -- Threshold: High-value customers defined as top 20% by revenue (80th percentile)
# MAGIC -- Output: sales_force_analysis temp view for territory optimization
# MAGIC -- Business Use: Identify top performers, workload imbalances, and coaching opportunities
# MAGIC -- Analyze sales force performance and customer distribution
# MAGIC CREATE OR REPLACE TEMP VIEW sales_force_analysis AS
# MAGIC WITH customer_revenues AS (
# MAGIC   SELECT CustomerID, SUM(TotalPrice) as customer_revenue
# MAGIC   FROM commercial.base.sales
# MAGIC   GROUP BY CustomerID
# MAGIC ),
# MAGIC high_value_threshold AS (
# MAGIC   SELECT PERCENTILE(customer_revenue, 0.8) as threshold
# MAGIC   FROM customer_revenues
# MAGIC )
# MAGIC SELECT 
# MAGIC   e.EmployeeID,
# MAGIC   e.FirstName || ' ' || e.LastName as SalesPersonName,
# MAGIC   COUNT(DISTINCT s.CustomerID) as customer_count,
# MAGIC   COUNT(DISTINCT s.SalesID) as transaction_count,
# MAGIC   SUM(s.TotalPrice) as total_revenue,
# MAGIC   AVG(s.TotalPrice) as avg_transaction_value,
# MAGIC   SUM(s.Quantity) as total_units_sold,
# MAGIC   -- Customer value distribution
# MAGIC   COUNT(DISTINCT CASE WHEN cr.customer_revenue >= (SELECT threshold FROM high_value_threshold) THEN s.CustomerID END) as high_value_customers,
# MAGIC   AVG(cr.customer_revenue) as avg_customer_value
# MAGIC FROM commercial.base.employees e
# MAGIC LEFT JOIN commercial.base.sales s ON e.EmployeeID = s.SalesPersonID
# MAGIC LEFT JOIN customer_revenues cr ON s.CustomerID = cr.CustomerID
# MAGIC GROUP BY e.EmployeeID, e.FirstName, e.LastName
# MAGIC HAVING COUNT(DISTINCT s.SalesID) > 0;
# MAGIC
# MAGIC SELECT * FROM sales_force_analysis ORDER BY total_revenue DESC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Sales Force Performance Tiers
# 🏆 SALES FORCE TIERING: Performance Classification
# What: Classify each sales rep into performance tiers using revenue and efficiency metrics
# Classification Logic:
#   • Top Performer: Top 25% in both revenue and revenue-per-customer
#   • Strong: Above average in key metrics
#   • Good: Meeting expectations
#   • Developing: Below average, needs training/support
# Metrics Analyzed: Total revenue, avg transaction value, customer count, high-value customers
# Output: Performance distribution and tier-level statistics
# Business Use: Compensation decisions, training allocation, account rebalancing
# Analyze sales force and develop optimization strategies

sales_force = spark.sql("""
  SELECT 
    EmployeeID,
    SalesPersonName,
    customer_count,
    transaction_count,
    total_revenue,
    avg_transaction_value,
    high_value_customers,
    avg_customer_value
  FROM sales_force_analysis
""").toPandas()

print(f"Analyzing {len(sales_force)} sales representatives\n")

# Performance classification
sales_force['revenue_per_customer'] = sales_force['total_revenue'] / sales_force['customer_count']
sales_force['performance_score'] = (
    (sales_force['total_revenue'] / sales_force['total_revenue'].max()) * 0.5 +
    (sales_force['high_value_customers'] / sales_force['high_value_customers'].max()) * 0.3 +
    (sales_force['avg_transaction_value'] / sales_force['avg_transaction_value'].max()) * 0.2
)

sales_force['performance_tier'] = pd.qcut(
    sales_force['performance_score'],
    q=4,
    labels=['Developing', 'Good', 'Strong', 'Top Performer'],
    duplicates='drop'
)

print("=== Sales Force Performance Distribution ===")
print(sales_force['performance_tier'].value_counts())

print("\n=== Performance Metrics by Tier ===")
perf_by_tier = sales_force.groupby('performance_tier').agg({
    'EmployeeID': 'count',
    'total_revenue': ['mean', 'sum'],
    'customer_count': 'mean',
    'high_value_customers': 'mean'
}).round(2)
print(perf_by_tier)

# COMMAND ----------

# DBTITLE 1,Sales Force Recommendations
# 💡 SALES FORCE OPTIMIZATION PLAN: Tier-Specific Actions
# What: Develop specific strategies for each performance tier
# Strategies:
#   • Top Performers: Retain with bonuses, assign largest accounts, mentor others
#   • Strong: Growth opportunities, stretch assignments, leadership development
#   • Good: Maintain standards, skill development, cross-training
#   • Developing: Structured training, coaching, performance improvement plans
# Includes: Revenue contribution, recommended actions, investment per tier
# Output: Actionable sales management plan with accountability metrics
# Business Use: Sales leadership roadmap for team optimization and territory planning
# Generate sales force optimization recommendations

print(f"\n{'='*80}")
print("SALES FORCE OPTIMIZATION RECOMMENDATIONS")
print(f"{'='*80}\n")

# Top Performers
top_performers = sales_force[sales_force['performance_tier'] == 'Top Performer']
print("\n🏆 TOP PERFORMERS - Retain & Leverage")
print("-" * 80)
print(f"Count: {len(top_performers)} reps")
print(f"Revenue: ${top_performers['total_revenue'].sum():,.2f} ({top_performers['total_revenue'].sum() / sales_force['total_revenue'].sum() * 100:.1f}% of total)")
print(f"Avg High-Value Customers: {top_performers['high_value_customers'].mean():.1f}")
print("\nTop 5:")
for idx, row in top_performers.nlargest(5, 'total_revenue').iterrows():
    print(f"  • {row['SalesPersonName']:30s} - ${row['total_revenue']:>12,.2f} | {row['customer_count']:.0f} customers")
print("\nRecommendations:")
print("  1. Retention bonuses and career development paths")
print("  2. Assign to highest-value accounts and prospects")
print("  3. Mentorship roles for developing reps")
print("  4. Share best practices in team meetings")
print("  5. Competitive compensation to prevent attrition")

# Developing Reps
developing = sales_force[sales_force['performance_tier'] == 'Developing']
if len(developing) > 0:
    print("\n\n📚 DEVELOPING REPS - Training & Support")
    print("-" * 80)
    print(f"Count: {len(developing)} reps")
    print(f"Revenue: ${developing['total_revenue'].sum():,.2f}")
    print(f"Avg Customers: {developing['customer_count'].mean():.1f}")
    print("\nRecommendations:")
    print("  1. Structured training program (product knowledge, sales techniques)")
    print("  2. Pair with top performers for shadowing")
    print("  3. Focus on transactional accounts (lower complexity)")
    print("  4. Weekly coaching sessions with sales manager")
    print("  5. Clear performance improvement plans with 90-day milestones")
    print(f"  6. Potential revenue if brought to 'Good' tier: ${(developing['customer_count'] * sales_force[sales_force['performance_tier'] == 'Good']['revenue_per_customer'].mean()).sum():,.2f}")

# Account Reallocation
print("\n\n🔄 ACCOUNT ALLOCATION OPTIMIZATION")
print("-" * 80)
total_high_value = sales_force['high_value_customers'].sum()
avg_high_value_per_top = top_performers['high_value_customers'].mean()

print(f"Total High-Value Customers: {total_high_value:.0f}")
print(f"Currently with Top Performers: {top_performers['high_value_customers'].sum():.0f} ({top_performers['high_value_customers'].sum() / total_high_value * 100:.1f}%)")
print("\nRecommendations:")
print("  1. Audit account assignments quarterly")
print("  2. Reallocate 80% of high-value accounts to top performers")
print("  3. Strong performers handle mid-tier accounts with growth potential")
print("  4. Developing reps focus on smaller transactional accounts")
print("  5. Implement territory management software for optimization")

# Compensation Alignment
print("\n\n💰 COMPENSATION STRUCTURE RECOMMENDATIONS")
print("-" * 80)
print("Current Issue: Performance varies significantly across team")
print("\nRecommended Structure:")
print("  • Base Salary: 60% (ensure stability)")
print("  • Commission: 30% (revenue-based, tiered for high-value sales)")
print("  • Bonuses: 10% (customer retention, new high-value acquisitions)")
print("\nSpecial Incentives:")
print("  • Accelerator for sales above quota (1.5x commission rate)")
print("  • Bonus for converting at-risk customers")
print("  • Team bonus for overall revenue targets")
print("  • SPIFFs for focus products (Stars category)")

# Territory Design
print("\n\n🗺️ TERRITORY OPTIMIZATION")
print("-" * 80)
print("Current State: Review customer distribution and sales coverage")
print("\nRecommendations:")
print("  1. Geographic clustering to reduce travel time")
print("  2. Balance territories by customer value (not just count)")
print("  3. Named accounts for top 100 customers")
print("  4. Inside sales team for small transactional customers")
print("  5. Specialized reps for key categories or verticals")
print(f"\nExpected Impact: 10-15% productivity improvement = ${sales_force['total_revenue'].sum() * 0.125:,.2f} additional revenue")

# COMMAND ----------

# DBTITLE 1,Summary Section
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 5. Executive Summary & Implementation Roadmap

# COMMAND ----------

# DBTITLE 1,Consolidated ROI Summary
# 📊 FINANCIAL IMPACT SUMMARY: Consolidated ROI Analysis
# What: Calculate total business impact across all 5 recommendation areas
# Components:
#   1. Inventory Optimization: Working capital freed
#   2. Customer Retention: Revenue from prevented churn and growth campaigns
#   3. Product Portfolio: Revenue from star products and margin improvements
#   4. Sales Force: Productivity gains from optimized territory allocation
#   5. Operational Efficiency: Cost reductions and process improvements
# Output: Total expected benefit, total investment, net value, and ROI multiple
# Business Use: Executive decision support - quantifies total business case
# Calculate total impact of all recommendations

print(f"\n{'='*80}")
print("PRESCRIPTIVE RECOMMENDATIONS - ROI SUMMARY")
print(f"{'='*80}\n")

# Current baseline (approximations based on analyses)
try:
    current_revenue = customer_data['total_revenue'].sum()
    current_inventory_investment = demand_data['reorder_point_value'].sum()
except:
    current_revenue = 4_000_000_000  # $4B from CLV analysis
    current_inventory_investment = 50_000_000  # $50M estimated

print(f"CURRENT STATE BASELINE")
print("-" * 80)
print(f"Annual Revenue: ${current_revenue:,.2f}")
print(f"Inventory Investment: ${current_inventory_investment:,.2f}")
print(f"Customer Base: {len(customer_data):,} customers" if 'customer_data' in dir() else "Customer Base: ~100,000 customers")
print(f"Product Portfolio: {len(product_perf):,} products" if 'product_perf' in dir() else "Product Portfolio: ~450 products")
print(f"Sales Force: {len(sales_force)} representatives" if 'sales_force' in dir() else "Sales Force: Active team")

print(f"\n\n{'='*80}")
print("EXPECTED FINANCIAL IMPACT (12-MONTH)")
print(f"{'='*80}\n")

# Initiative impacts
impact_summary = []

# 1. Inventory Optimization
inv_savings = current_inventory_investment * 0.15  # 15% reduction
impact_summary.append({
    'Initiative': 'Inventory Optimization',
    'Type': 'Cost Savings',
    'Impact': inv_savings,
    'Investment': current_inventory_investment * 0.02,  # 2% for systems/process
    'Timeline': '3-6 months'
})

# 2. Customer Retention
retention_revenue = current_revenue * 0.08  # 8% from retention campaigns
impact_summary.append({
    'Initiative': 'Customer Retention Programs',
    'Type': 'Revenue Increase',
    'Impact': retention_revenue,
    'Investment': len(customer_data) * 15 if 'customer_data' in dir() else 1_500_000,
    'Timeline': '6-9 months'
})

# 3. CLV Growth
clv_growth = current_revenue * 0.10  # 10% from CLV optimization
impact_summary.append({
    'Initiative': 'CLV Growth Strategies',
    'Type': 'Revenue Increase',
    'Impact': clv_growth,
    'Investment': len(customer_data) * 20 if 'customer_data' in dir() else 2_000_000,
    'Timeline': '9-12 months'
})

# 4. Product Portfolio
portfolio_impact = current_revenue * 0.05  # 5% from product optimization
impact_summary.append({
    'Initiative': 'Product Portfolio Optimization',
    'Type': 'Revenue Increase',
    'Impact': portfolio_impact,
    'Investment': 500_000,  # Marketing, clearance costs
    'Timeline': '6-12 months'
})

# 5. Sales Force
sales_force_impact = current_revenue * 0.12  # 12% from sales optimization
impact_summary.append({
    'Initiative': 'Sales Force Optimization',
    'Type': 'Revenue Increase',
    'Impact': sales_force_impact,
    'Investment': 800_000,  # Training, compensation changes
    'Timeline': '3-9 months'
})

# Display summary
for item in impact_summary:
    print(f"{item['Initiative']}")
    print("-" * 80)
    print(f"  Type: {item['Type']}")
    print(f"  Expected Impact: ${item['Impact']:,.2f}")
    print(f"  Required Investment: ${item['Investment']:,.2f}")
    print(f"  ROI: {item['Impact'] / item['Investment']:.1f}x")
    print(f"  Timeline: {item['Timeline']}")
    print()

# Total impact
total_impact = sum([item['Impact'] for item in impact_summary])
total_investment = sum([item['Investment'] for item in impact_summary])

print(f"\n{'='*80}")
print("TOTAL PROGRAM IMPACT")
print(f"{'='*80}")
print(f"Total Expected Impact: ${total_impact:,.2f}")
print(f"Total Investment Required: ${total_investment:,.2f}")
print(f"Overall ROI: {total_impact / total_investment:.1f}x")
print(f"Impact as % of Revenue: {total_impact / current_revenue * 100:.1f}%")
print(f"\nNet Benefit (Year 1): ${total_impact - total_investment:,.2f}")

# COMMAND ----------

# DBTITLE 1,Implementation Roadmap
# 📅 90-DAY IMPLEMENTATION ROADMAP: Phased Rollout Plan
# What: Detailed 3-phase implementation plan with week-by-week activities
# Phases:
#   • Phase 1 (Days 1-30): Assessment, planning, and quick wins
#   • Phase 2 (Days 31-60): Foundation building, systems deployment, scaling
#   • Phase 3 (Days 61-90): Full deployment, review, and refinement
# Includes: Success metrics and KPIs for tracking progress across all areas
# Metrics: Inventory turnover, retention rate, revenue per rep, ROI, etc.
# Business Use: Program management roadmap with clear milestones and accountability
# 90-Day implementation roadmap

print(f"\n{'='*80}")
print("90-DAY IMPLEMENTATION ROADMAP")
print(f"{'='*80}\n")

print("PHASE 1: IMMEDIATE ACTIONS (Days 1-30)")
print("-" * 80)
print("Week 1-2: Assessment & Planning")
print("  • Executive approval and budget allocation")
print("  • Form cross-functional implementation team")
print("  • Detailed data validation and baseline metrics")
print("  • Vendor selection for inventory management system")

print("\nWeek 3-4: Quick Wins")
print("  • Launch win-back campaign for high-value at-risk customers")
print("  • Implement reorder recommendations for A-class items")
print("  • Begin clearance promotion for discontinuation candidates")
print("  • Sales team briefing on account reallocation plan")

print("\n\nPHASE 2: FOUNDATION BUILDING (Days 31-60)")
print("-" * 80)
print("Week 5-6: Systems & Process")
print("  • Deploy inventory management dashboards")
print("  • Launch VIP loyalty program for active high-value customers")
print("  • Begin sales training program for developing reps")
print("  • Implement automated reorder triggers for critical items")

print("\nWeek 7-8: Scaling")
print("  • Roll out customer segmentation to all customer-facing teams")
print("  • Expand product performance monitoring")
print("  • Finalize territory realignment")
print("  • Measure early results and adjust tactics")

print("\n\nPHASE 3: OPTIMIZATION & SCALE (Days 61-90)")
print("-" * 80)
print("Week 9-10: Full Deployment")
print("  • Complete account reallocation to sales force")
print("  • Launch growth campaigns for medium CLV customers")
print("  • Finalize product portfolio decisions")
print("  • Implement new compensation structure")

print("\nWeek 11-12: Review & Refine")
print("  • 90-day performance review against targets")
print("  • Customer and employee feedback collection")
print("  • Refine strategies based on learnings")
print("  • Plan for next 90 days")

print(f"\n\n{'='*80}")
print("SUCCESS METRICS & KPIs")
print(f"{'='*80}\n")

print("Track these KPIs monthly:")
print("\nInventory Metrics:")
print("  • Inventory turnover ratio (Target: +20%)")
print("  • Stock-out rate (Target: <2%)")
print("  • Working capital tied in inventory (Target: -15%)")
print("  • Obsolete inventory value (Target: -50%)")

print("\nCustomer Metrics:")
print("  • Customer retention rate (Target: 90%+)")
print("  • High-value customer retention (Target: 95%+)")
print("  • Average customer lifetime value (Target: +10%)")
print("  • Win-back campaign success rate (Target: 35%+)")

print("\nProduct Metrics:")
print("  • Revenue from star products (Target: +15%)")
print("  • Return rate reduction (Target: -20%)")
print("  • Products discontinued (Target: 50-100 items)")
print("  • New product success rate (Target: >60%)")

print("\nSales Force Metrics:")
print("  • Revenue per sales rep (Target: +12%)")
print("  • Average deal size (Target: +8%)")
print("  • High-value accounts per top performer (Target: 15-20)")
print("  • Sales rep satisfaction score (Target: >4.0/5.0)")

print("\nFinancial Metrics:")
print(f"  • Revenue growth (Target: {total_impact / current_revenue * 100:.1f}%)")
print(f"  • Operating margin improvement (Target: 2-3 points)")
print(f"  • Return on investment (Target: {total_impact / total_investment:.1f}x)")
print(f"  • Cash flow improvement (Target: ${inv_savings:,.0f})")

# COMMAND ----------

# DBTITLE 1,Final Recommendations Summary
# 🎯 EXECUTIVE SUMMARY: Prescriptive Recommendations Overview
# What: Consolidated one-page summary of all recommendations for executive presentation
# Structure:
#   • Key Recommendations: 5 strategic areas with specific actions
#   • Bottom Line Impact: Total benefit, investment, net value, ROI
#   • Implementation Timeline: 90-day rollout with quick wins
#   • Next Steps: Immediate actions for getting started
# Output: Executive-ready summary suitable for board presentation or strategic planning
# Business Use: Decision document for executive approval and resource allocation
# Final summary for stakeholders

print(f"\n{'='*80}")
print("PRESCRIPTIVE RECOMMENDATIONS - EXECUTIVE SUMMARY")
print(f"{'='*80}\n")

print("This analysis transforms predictive insights into actionable strategies across")
print("five key operational areas, with quantified business impact and clear ROI.")

print(f"\n\n🎯 KEY RECOMMENDATIONS:")
print("-" * 80)

print("\n1. INVENTORY OPTIMIZATION")
print("   • Implement automated reorder system for A-class items")
print("   • Reduce safety stock for C-class by 20%")
print("   • Adopt JIT delivery for perishable products")
print(f"   • Impact: ${inv_savings:,.0f} working capital freed")

print("\n2. CUSTOMER RETENTION & GROWTH")
print("   • Launch 4 targeted campaigns by customer segment")
print("   • VIP loyalty program for top 20% of customers")
print("   • Win-back high-value at-risk customers (immediate)")
print(f"   • Impact: ${retention_revenue + clv_growth:,.0f} revenue increase")

print("\n3. PRODUCT PORTFOLIO OPTIMIZATION")
print("   • Maximize investment in 'Star' products")
print("   • Phase out 50-100 underperforming SKUs")
print("   • Address quality issues in high-return products")
print(f"   • Impact: ${portfolio_impact:,.0f} revenue + improved margins")

print("\n4. SALES FORCE OPTIMIZATION")
print("   • Reallocate accounts by customer value")
print("   • Structured training for developing reps")
print("   • New compensation structure aligned with value")
print(f"   • Impact: ${sales_force_impact:,.0f} productivity gain")

print("\n5. OPERATIONAL EFFICIENCY")
print("   • Deploy integrated analytics dashboards")
print("   • Automate routine processes")
print("   • Monthly performance reviews")
print(f"   • Impact: Cross-functional visibility and agility")

print(f"\n\n📈 BOTTOM LINE IMPACT:")
print("-" * 80)
print(f"Total Expected Benefit: ${total_impact:,.2f}")
print(f"Total Investment: ${total_investment:,.2f}")
print(f"Net Value (Year 1): ${total_impact - total_investment:,.2f}")
print(f"ROI: {total_impact / total_investment:.1f}x")
print(f"Revenue Impact: {total_impact / current_revenue * 100:.1f}% increase")

print(f"\n\n⏰ IMPLEMENTATION:")
print("-" * 80)
print("90-day fast-start with phased rollout")
print("Quick wins in first 30 days")
print("Full deployment by end of quarter")
print("Measurable results within 6 months")

print(f"\n\n✅ NEXT STEPS:")
print("-" * 80)
print("1. Executive review and approval (Week 1)")
print("2. Detailed project planning and team formation (Week 2)")
print("3. Begin Phase 1 quick wins (Week 3-4)")
print("4. Monthly steering committee reviews")
print("5. Quarterly performance assessment against targets")

print(f"\n{'='*80}")
print("END OF PRESCRIPTIVE RECOMMENDATIONS")
print(f"{'='*80}\n")

print("Next: Move to ROI Analysis notebook for detailed financial modeling")

# COMMAND ----------

# DBTITLE 1,📋 MBA Formula Validation & Corrections
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 📋 MBA Academic Formula Validation & Corrections
# MAGIC
# MAGIC ### Purpose
# MAGIC This section validates all calculations against MBA textbook standards and documents any deviations.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ CORRECT FORMULAS (No Changes Needed)
# MAGIC
# MAGIC ### 1. Safety Stock Formula (Cell 5)
# MAGIC **Current Implementation:**
# MAGIC ```
# MAGIC Safety Stock = Z × √(Lead Time) × Demand Std Dev
# MAGIC ```
# MAGIC
# MAGIC **Academic Standard:**
# MAGIC ```
# MAGIC SS = Z × σ_LT
# MAGIC where σ_LT = σ_D × √L
# MAGIC ```
# MAGIC
# MAGIC ✅ **Status: CORRECT**
# MAGIC - Matches textbook formula from Chopra & Meindl (2015)
# MAGIC - Z = 1.65 for 95% service level is standard
# MAGIC - Uses √(Lead Time) correctly for demand uncertainty
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. Reorder Point (ROP) Formula (Cell 5)
# MAGIC **Current Implementation:**
# MAGIC ```
# MAGIC ROP = (Average Daily Demand × Lead Time) + Safety Stock
# MAGIC ```
# MAGIC
# MAGIC **Academic Standard:**
# MAGIC ```
# MAGIC ROP = d × L + SS
# MAGIC where:
# MAGIC   d = average demand per period
# MAGIC   L = lead time
# MAGIC   SS = safety stock
# MAGIC ```
# MAGIC
# MAGIC ✅ **Status: CORRECT**
# MAGIC - Standard operations management formula
# MAGIC - Source: Heizer & Render (2017), "Operations Management"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Customer Retention ROI (Cells 12-13)
# MAGIC **Current Implementation:**
# MAGIC ```
# MAGIC Retention ROI = (Recovered Revenue - Campaign Cost) / Campaign Cost × 100%
# MAGIC
# MAGIC Assumptions:
# MAGIC - High-value recovery rate: 30-40%
# MAGIC - Loyalty program lift: 15%
# MAGIC - Campaign costs: $15-50 per customer by tier
# MAGIC ```
# MAGIC
# MAGIC ✅ **Status: CORRECT**
# MAGIC - Industry-standard ROI calculation
# MAGIC - Recovery rates align with marketing benchmarks (Reichheld, 2003)
# MAGIC - Investment levels are realistic for segmented campaigns
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚠️ ISSUES REQUIRING CORRECTION
# MAGIC
# MAGIC ### 4. Order Quantity Calculation (Cell 5)
# MAGIC **Current Implementation:**
# MAGIC ```python
# MAGIC order_quantity = avg_daily_demand × review_period_days  # 14 days
# MAGIC ```
# MAGIC
# MAGIC ❌ **Issue: This is NOT EOQ (Economic Order Quantity)**
# MAGIC
# MAGIC **What We're Actually Using:**
# MAGIC - Review-Period Ordering System (R-system)
# MAGIC - Orders a fixed amount of days' worth of demand
# MAGIC - Simpler but NOT cost-optimized
# MAGIC
# MAGIC **True EOQ Formula:**
# MAGIC ```
# MAGIC EOQ = √[(2 × D × S) / H]
# MAGIC
# MAGIC where:
# MAGIC   D = Annual demand (units)
# MAGIC   S = Ordering cost per order ($)
# MAGIC   H = Holding cost per unit per year ($)
# MAGIC ```
# MAGIC
# MAGIC **Why It Matters:**
# MAGIC - EOQ minimizes total inventory costs (ordering + holding)
# MAGIC - Review-period ordering may order too much or too little
# MAGIC - For MBA exam: Must know true EOQ formula
# MAGIC
# MAGIC **Correction Needed:**
# MAGIC Either:
# MAGIC 1. Rename to "Review-Period Order Quantity" (honest about what it is), OR
# MAGIC 2. Implement true EOQ with ordering/holding costs
# MAGIC
# MAGIC **Academic Source:** Harris (1913), Wilson (1934) - Original EOQ model
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. ABC Classification (Cell 6)
# MAGIC **Current Implementation:**
# MAGIC ```python
# MAGIC demand_data['abc_class'] = pd.qcut(
# MAGIC     demand_data['inventory_value'], 
# MAGIC     q=3,  # Equal tertiles
# MAGIC     labels=['C', 'B', 'A']
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ❌ **Issue: Doesn't Follow Pareto 80/20 Principle**
# MAGIC
# MAGIC **Current Approach:**
# MAGIC - Divides products into 3 equal groups (33% each)
# MAGIC - A = top 33%, B = middle 33%, C = bottom 33%
# MAGIC
# MAGIC **True ABC Classification (Pareto):**
# MAGIC ```
# MAGIC A-class: Top 70-80% of value (typically 20% of items)
# MAGIC B-class: Next 15-20% of value (typically 30% of items)
# MAGIC C-class: Remaining 5-10% of value (typically 50% of items)
# MAGIC ```
# MAGIC
# MAGIC **Correct Implementation:**
# MAGIC ```python
# MAGIC # Calculate cumulative value percentage
# MAGIC demand_data = demand_data.sort_values('inventory_value', ascending=False)
# MAGIC demand_data['cumulative_value_pct'] = (
# MAGIC     demand_data['inventory_value'].cumsum() / 
# MAGIC     demand_data['inventory_value'].sum() * 100
# MAGIC )
# MAGIC
# MAGIC # True ABC classification
# MAGIC def classify_abc_pareto(cum_pct):
# MAGIC     if cum_pct <= 80:   # First 80% of value
# MAGIC         return 'A'
# MAGIC     elif cum_pct <= 95:  # Next 15% of value
# MAGIC         return 'B'
# MAGIC     else:               # Last 5% of value
# MAGIC         return 'C'
# MAGIC
# MAGIC demand_data['abc_class'] = demand_data['cumulative_value_pct'].apply(classify_abc_pareto)
# MAGIC ```
# MAGIC
# MAGIC **Why It Matters:**
# MAGIC - ABC is based on Pareto principle: 20% of items = 80% of value
# MAGIC - Equal tertiles don't capture this insight
# MAGIC - Management focus should be on the vital few (A-items)
# MAGIC
# MAGIC **Academic Source:** 
# MAGIC - Pareto, V. (1896) - Pareto Principle
# MAGIC - Dickie, H.F. (1951) - ABC inventory analysis
# MAGIC - Flores & Whybark (1987) - "Implementing Multiple Criteria ABC Analysis"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 6. Total ROI Calculation (Cell 23)
# MAGIC **Current Implementation:**
# MAGIC ```python
# MAGIC Total Expected Impact: $1.42 billion
# MAGIC Total Investment: $4.8 million
# MAGIC Overall ROI: 295.7x (29,570%)
# MAGIC ```
# MAGIC
# MAGIC ⚠️ **Issue: ROI Appears Unrealistically High**
# MAGIC
# MAGIC **Analysis:**
# MAGIC - 295× return is extraordinary (venture capital targets 10-20×)
# MAGIC - Implies every $1 invested returns $296
# MAGIC - Such returns are rare in operational improvements
# MAGIC
# MAGIC **Why This Happened:**
# MAGIC - Revenue increases are counted as 100% benefit
# MAGIC - Doesn't account for:
# MAGIC   * Cost of goods sold (COGS)
# MAGIC   * Margin rates (typically 20-40% in retail/wholesale)
# MAGIC   * Implementation challenges
# MAGIC   * Probability of success (not all initiatives succeed)
# MAGIC
# MAGIC **Correct Approach:**
# MAGIC ```
# MAGIC Benefit = Revenue Increase × Margin Rate × Success Probability
# MAGIC
# MAGIC Example:
# MAGIC - Revenue increase: $100M
# MAGIC - Margin rate: 30%
# MAGIC - Success probability: 70%
# MAGIC - True benefit: $100M × 0.30 × 0.70 = $21M
# MAGIC
# MAGIC ROI = ($21M - $5M investment) / $5M = 3.2× (320%)
# MAGIC ```
# MAGIC
# MAGIC **More Realistic ROI Targets:**
# MAGIC - Inventory optimization: 3-5× ROI
# MAGIC - Customer retention: 5-10× ROI
# MAGIC - Sales force: 2-4× ROI
# MAGIC - Product portfolio: 3-6× ROI
# MAGIC - Overall program: 4-8× ROI is excellent
# MAGIC
# MAGIC **Correction Needed:**
# MAGIC Adjust calculations to use margin rates, not gross revenue.
# MAGIC
# MAGIC **Academic Source:** 
# MAGIC - Kaplan & Norton (2008) - "The Execution Premium"
# MAGIC - Davenport & Harris (2007) - "Competing on Analytics"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 SUMMARY FOR VIVA PREPARATION
# MAGIC
# MAGIC ### Formulas to Memorize (CORRECT in this notebook):
# MAGIC 1. **Safety Stock:** `SS = Z × σ_D × √L`
# MAGIC 2. **Reorder Point:** `ROP = d × L + SS`
# MAGIC 3. **Retention ROI:** `(Recovered CLV - Cost) / Cost`
# MAGIC
# MAGIC ### Formulas to Learn (NOT in this notebook):
# MAGIC 1. **EOQ:** `EOQ = √[(2 × D × S) / H]`
# MAGIC 2. **ABC (Pareto):** A = top 80% value, B = next 15%, C = last 5%
# MAGIC 3. **Adjusted ROI:** `(Benefit × Margin × Probability - Investment) / Investment`
# MAGIC
# MAGIC ### What to Say in Viva:
# MAGIC - "We used review-period ordering for simplicity, but true EOQ would be more cost-optimal"
# MAGIC - "ABC uses tertiles for balanced classification, but Pareto-based ABC (80/20) is the academic standard"
# MAGIC - "ROI calculations assume gross revenue; adjusting for margins would give more conservative estimates"
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ RECOMMENDATION
# MAGIC
# MAGIC **For MBA Exam/Viva:**
# MAGIC 1. Know the CORRECT formulas (EOQ, Pareto ABC, margin-adjusted ROI)
# MAGIC 2. Acknowledge the simplifications in this notebook
# MAGIC 3. Explain when simplifications are acceptable vs. when full models are needed
# MAGIC
# MAGIC **For Real Business Implementation:**
# MAGIC - The current formulas are pragmatic and workable
# MAGIC - Review-period ordering is common in practice
# MAGIC - Simplified ABC is easier to communicate to management
# MAGIC - ROI calculations should use margins for financial reporting
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,✅ CORRECTED: True EOQ Calculation
# ✅ CORRECTED FORMULA: Economic Order Quantity (EOQ)
# What: Calculate TRUE optimal order quantity using EOQ formula
# Formula: EOQ = √[(2 × D × S) / H]
# Where:
#   D = Annual demand (units/year)
#   S = Ordering cost per order ($)
#   H = Holding cost per unit per year ($/unit/year)
# Source: Harris (1913), Wilson (1934) - Original EOQ model

print("\n" + "="*80)
print("✅ CORRECTED: TRUE EOQ CALCULATION (MBA Standard Formula)")
print("="*80)

# Assumptions for EOQ calculation
ordering_cost_per_order = 50  # $ per order (processing, shipping, labor)
holding_cost_rate = 0.25  # 25% of unit cost per year (capital, storage, insurance, obsolescence)

# Calculate EOQ for each product
demand_data_copy = demand_data.copy()
demand_data_copy['annual_demand'] = demand_data_copy['avg_daily_demand'] * 365
demand_data_copy['holding_cost_per_unit'] = demand_data_copy['Price'] * holding_cost_rate

# True EOQ formula: sqrt((2 * D * S) / H)
demand_data_copy['eoq_true'] = np.sqrt(
    (2 * demand_data_copy['annual_demand'] * ordering_cost_per_order) / 
    demand_data_copy['holding_cost_per_unit']
).fillna(0).round(0)

# Calculate total annual costs at EOQ
demand_data_copy['orders_per_year_eoq'] = (
    demand_data_copy['annual_demand'] / demand_data_copy['eoq_true']
).replace([np.inf, -np.inf], 0)

demand_data_copy['annual_ordering_cost'] = (
    demand_data_copy['orders_per_year_eoq'] * ordering_cost_per_order
)

demand_data_copy['average_inventory_eoq'] = demand_data_copy['eoq_true'] / 2
demand_data_copy['annual_holding_cost'] = (
    demand_data_copy['average_inventory_eoq'] * demand_data_copy['holding_cost_per_unit']
)

demand_data_copy['total_inventory_cost'] = (
    demand_data_copy['annual_ordering_cost'] + demand_data_copy['annual_holding_cost']
)

print("\n=== EOQ vs Review-Period Ordering Comparison ===")
print(f"\nSample Products (First 5):\n")

comparison = demand_data_copy[[
    'ProductName', 'avg_daily_demand', 'Price',
    'order_quantity', 'eoq_true', 'total_inventory_cost'
]].head(5)

for idx, row in comparison.iterrows():
    print(f"\n{row['ProductName'][:40]:40s}")
    print(f"  Daily Demand: {row['avg_daily_demand']:.1f} units")
    print(f"  Price: ${row['Price']:.2f}")
    print(f"  Review-Period Qty (14 days): {row['order_quantity']:.0f} units")
    print(f"  TRUE EOQ: {row['eoq_true']:.0f} units")
    print(f"  Annual Inventory Cost (at EOQ): ${row['total_inventory_cost']:.2f}")
    print(f"  Difference: {row['eoq_true'] - row['order_quantity']:.0f} units")

print(f"\n\n=== Key Insight (EOQ Property) ===")
print("At optimal EOQ:")
print(f"  Average Ordering Cost = Average Holding Cost")
print(f"\nExample verification (Product 1):")
if len(demand_data_copy) > 0:
    sample = demand_data_copy.iloc[0]
    print(f"  Ordering Cost: ${sample['annual_ordering_cost']:.2f}")
    print(f"  Holding Cost: ${sample['annual_holding_cost']:.2f}")
    print(f"  Ratio: {sample['annual_ordering_cost'] / sample['annual_holding_cost']:.2f} (should be ~1.0)")

print(f"\n\n=== Total Inventory Cost Savings (Using EOQ) ===")
total_current_cost = demand_data_copy['total_inventory_cost'].sum()
print(f"Total Annual Inventory Cost (at EOQ): ${total_current_cost:,.2f}")
print(f"\nNote: Compare this to actual costs to calculate EOQ savings")
print(f"Typical savings from EOQ optimization: 10-30% of inventory costs")

print("\n" + "="*80)
print("✅ MBA EXAM TIP: Memorize EOQ = √[(2DS)/H] and its properties!")
print("="*80)

# COMMAND ----------

# DBTITLE 1,✅ CORRECTED: Pareto-Based ABC
# ✅ CORRECTED FORMULA: ABC Classification Using Pareto Principle
# What: Classify using cumulative value (80/20 rule)
# A = top 80% value, B = next 15%, C = last 5%

print("\n" + "="*80)
print("✅ CORRECTED: PARETO-BASED ABC CLASSIFICATION")
print("="*80)

demand_sorted = demand_data.sort_values('inventory_value', ascending=False).copy()
total_value = demand_sorted['inventory_value'].sum()
demand_sorted['cumulative_value'] = demand_sorted['inventory_value'].cumsum()
demand_sorted['cumulative_value_pct'] = (demand_sorted['cumulative_value'] / total_value * 100)

def classify_abc_pareto(cum_pct):
    if cum_pct <= 80: return 'A'
    elif cum_pct <= 95: return 'B'
    else: return 'C'

demand_sorted['abc_pareto'] = demand_sorted['cumulative_value_pct'].apply(classify_abc_pareto)

abc_summary = demand_sorted.groupby('abc_pareto').agg({
    'ProductID': 'count',
    'inventory_value': 'sum'
})
total_items = len(demand_sorted)
abc_summary['Item_Pct'] = (abc_summary['ProductID'] / total_items * 100).round(1)
abc_summary['Value_Pct'] = (abc_summary['inventory_value'] / total_value * 100).round(1)

print("\n=== Pareto ABC Analysis ===")
for abc in ['A', 'B', 'C']:
    row = abc_summary.loc[abc]
    print(f"\n{abc}-Class:")
    print(f"  Items: {row['Item_Pct']:.1f}% ({int(row['ProductID'])} products)")
    print(f"  Value: {row['Value_Pct']:.1f}% (${row['inventory_value']:,.2f})")

print("\n" + "="*80)
print("✅ MBA TIP: ABC follows Pareto 80/20, not equal splits!")
print("="*80)

# COMMAND ----------

# DBTITLE 1,✅ CORRECTED: Margin-Adjusted ROI
# ✅ CORRECTED FORMULA: Margin-Adjusted ROI
# What: Apply margin rates and success probability to revenue increases
# Formula: Benefit = Revenue × Margin × Probability

print("\n" + "="*80)
print("✅ CORRECTED: MARGIN-ADJUSTED ROI")
print("="*80)

avg_margin = 0.30  # 30% net margin
success_prob = 0.70  # 70% probability

print(f"\nAssumptions: {avg_margin*100:.0f}% margin, {success_prob*100:.0f}% success probability")

# Adjusted benefits
inv_benefit = inv_savings * success_prob
ret_benefit = retention_revenue * avg_margin * success_prob
clv_benefit = clv_growth * avg_margin * success_prob  
port_benefit = portfolio_impact * avg_margin * success_prob
sales_benefit = sales_force_impact * avg_margin * success_prob

total_adjusted = inv_benefit + ret_benefit + clv_benefit + port_benefit + sales_benefit

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"\nORIGINAL (Gross Revenue):")
print(f"  Total: ${total_impact:,.2f}")
print(f"  ROI: {total_impact / total_investment:.1f}x")
print(f"  ⚠️ Unrealistic - no margins applied")

print(f"\nCORRECTED (Margin-Adjusted):")
print(f"  Total: ${total_adjusted:,.2f}")
print(f"  ROI: {total_adjusted / total_investment:.1f}x")
print(f"  ✅ Realistic for operational programs")

print(f"\n\n=== Industry Benchmarks ===")
print("Typical ROI for Operational Improvements:")
print("  • Inventory: 3-5x")
print("  • Retention: 5-10x")
print("  • Sales: 2-4x")
print(f"  • Overall: 4-8x is EXCELLENT")
print(f"\nOur Adjusted ROI: {total_adjusted / total_investment:.1f}x")
if total_adjusted / total_investment >= 4:
    print("✅ Within excellent range!")

print("\n" + "="*80)
print("✅ MBA TIP: Always adjust revenue for margins & probability!")
print("="*80)

# COMMAND ----------

