# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Project Overview
# MAGIC %md
# MAGIC # MBA Analytics & Operations Project - Descriptive Analytics
# MAGIC
# MAGIC ## Project Objectives
# MAGIC
# MAGIC This notebook implements comprehensive descriptive analytics for inventory management and customer analytics using the `commercial.base` schema.
# MAGIC
# MAGIC ### Covered Analyses:
# MAGIC 1. **Multi-Dimensional Inventory Classification**: ABC, FSN, VED, HML analyses
# MAGIC 2. **Customer Segmentation**: RFM analysis and behavioral segments
# MAGIC 3. **Return & Cancellation Analysis**: Product quality and operational insights
# MAGIC 4. **Product Portfolio Analysis**: Performance and profitability metrics
# MAGIC 5. **Geographic & Sales Force Analysis**: Regional and team performance
# MAGIC
# MAGIC ### Data Sources:
# MAGIC - `commercial.base.sales` - Transaction data
# MAGIC - `commercial.base.products` - Product master data
# MAGIC - `commercial.base.customers` - Customer information
# MAGIC - `commercial.base.categories` - Product categories
# MAGIC - `commercial.base.sales_cancelled_returned` - Returns/cancellations
# MAGIC - `commercial.base.employees` - Sales team data
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Data Quality Assessment
# MAGIC %md
# MAGIC ## 1. Data Quality & Validation Checks
# MAGIC
# MAGIC Before proceeding with analysis, we'll validate data completeness and identify any data quality issues.

# COMMAND ----------

# DBTITLE 1,Data Quality Metrics
# MAGIC %sql
# MAGIC -- Data Quality Assessment
# MAGIC SELECT 
# MAGIC   'sales' as table_name,
# MAGIC   COUNT(*) as total_records,
# MAGIC   COUNT(DISTINCT SalesID) as unique_sales,
# MAGIC   COUNT(DISTINCT CustomerID) as unique_customers,
# MAGIC   COUNT(DISTINCT ProductID) as unique_products,
# MAGIC   MIN(SalesDate) as earliest_date,
# MAGIC   MAX(SalesDate) as latest_date,
# MAGIC   DATEDIFF(MAX(SalesDate), MIN(SalesDate)) as days_of_data,
# MAGIC   SUM(CASE WHEN TotalPrice IS NULL THEN 1 ELSE 0 END) as null_prices,
# MAGIC   SUM(CASE WHEN Quantity IS NULL THEN 1 ELSE 0 END) as null_quantities
# MAGIC FROM commercial.base.sales
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'products' as table_name,
# MAGIC   COUNT(*) as total_records,
# MAGIC   COUNT(DISTINCT ProductID) as unique_products,
# MAGIC   COUNT(DISTINCT CategoryID) as unique_categories,
# MAGIC   0 as unused1,
# MAGIC   NULL as unused2,
# MAGIC   NULL as unused3,
# MAGIC   0 as unused4,
# MAGIC   SUM(CASE WHEN Price IS NULL THEN 1 ELSE 0 END) as null_prices,
# MAGIC   SUM(CASE WHEN VitalityDays IS NULL THEN 1 ELSE 0 END) as null_vitality
# MAGIC FROM commercial.base.products
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'customers' as table_name,
# MAGIC   COUNT(*) as total_records,
# MAGIC   COUNT(DISTINCT CustomerID) as unique_customers,
# MAGIC   COUNT(DISTINCT CityID) as unique_cities,
# MAGIC   0 as unused1,
# MAGIC   NULL as unused2,
# MAGIC   NULL as unused3,
# MAGIC   0 as unused4,
# MAGIC   SUM(CASE WHEN FirstName IS NULL THEN 1 ELSE 0 END) as null_names,
# MAGIC   0 as unused5
# MAGIC FROM commercial.base.customers
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'sales_cancelled_returned' as table_name,
# MAGIC   COUNT(*) as total_records,
# MAGIC   COUNT(DISTINCT SalesID) as unique_sales,
# MAGIC   COUNT(DISTINCT CustomerID) as unique_customers,
# MAGIC   COUNT(DISTINCT ProductID) as unique_products,
# MAGIC   MIN(CancellationDate) as earliest_date,
# MAGIC   MAX(CancellationDate) as latest_date,
# MAGIC   0 as unused1,
# MAGIC   SUM(CASE WHEN RefundAmount IS NULL THEN 1 ELSE 0 END) as null_refunds,
# MAGIC   SUM(CASE WHEN CancellationReason IS NULL THEN 1 ELSE 0 END) as null_reasons
# MAGIC FROM commercial.base.sales_cancelled_returned

# COMMAND ----------

# DBTITLE 1,Section 1 Header
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 2. Objective 1: Multi-Dimensional Inventory Classification
# MAGIC
# MAGIC ### 2.1 ABC Analysis (Value-Based Classification)
# MAGIC
# MAGIC Classify products based on cumulative revenue contribution:
# MAGIC - **A-items**: Top 70-80% of total value (high value, tight control)
# MAGIC - **B-items**: Next 15-25% of value (moderate control)
# MAGIC - **C-items**: Remaining 5-10% of value (minimal monitoring)

# COMMAND ----------

# DBTITLE 1,ABC Analysis - Theory & Preparation
# MAGIC %md
# MAGIC ### 📚 ABC Analysis - Theory & Viva Preparation
# MAGIC
# MAGIC #### **Definition & Origin**
# MAGIC ABC Analysis is based on the **Pareto Principle (80/20 rule)**, developed by Vilfredo Pareto. In inventory management, it suggests that ~80% of value comes from ~20% of items.
# MAGIC
# MAGIC #### **Core Concept**
# MAGIC - **Purpose**: Prioritize inventory control efforts based on value contribution
# MAGIC - **Classification**: Items are ranked by cumulative revenue/value contribution
# MAGIC   - **A-items**: Top 70-80% of total value (tight control, frequent monitoring)
# MAGIC   - **B-items**: Next 15-25% of value (moderate control)
# MAGIC   - **C-items**: Remaining 5-10% of value (minimal monitoring, relaxed control)
# MAGIC
# MAGIC #### **Methodology**
# MAGIC 1. Calculate total revenue for each product
# MAGIC 2. Sort products in descending order by revenue
# MAGIC 3. Calculate cumulative revenue percentage
# MAGIC 4. Classify based on cumulative thresholds
# MAGIC
# MAGIC #### **Business Application**
# MAGIC - **A-items**: Daily monitoring, tight inventory control, accurate demand forecasting, multiple suppliers
# MAGIC - **B-items**: Weekly/monthly review, moderate safety stock
# MAGIC - **C-items**: Annual review, simple reorder systems, bulk purchasing
# MAGIC
# MAGIC #### **Advantages**
# MAGIC - Focuses management attention on critical items
# MAGIC - Reduces inventory holding costs
# MAGIC - Improves cash flow management
# MAGIC - Simplifies inventory decision-making
# MAGIC
# MAGIC #### **Limitations**
# MAGIC - Only considers monetary value (ignores criticality, lead time, availability)
# MAGIC - Static analysis (doesn't account for demand variability)
# MAGIC - May overlook low-value but critical items
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,ABC Analysis Calculation
# MAGIC %sql
# MAGIC -- ABC Analysis: Value-based product classification
# MAGIC CREATE OR REPLACE TEMP VIEW abc_analysis AS
# MAGIC WITH product_revenue AS (
# MAGIC   SELECT 
# MAGIC     s.ProductID,
# MAGIC     p.ProductName,
# MAGIC     p.CategoryID,
# MAGIC     c.CategoryName,
# MAGIC     SUM(s.Quantity * p.Price) as total_revenue,
# MAGIC     SUM(s.Quantity) as total_quantity_sold,
# MAGIC     COUNT(DISTINCT s.SalesID) as transaction_count
# MAGIC   FROM commercial.base.sales s
# MAGIC   JOIN commercial.base.products p ON s.ProductID = p.ProductID
# MAGIC   LEFT JOIN commercial.base.categories c ON p.CategoryID = c.CategoryID
# MAGIC   GROUP BY s.ProductID, p.ProductName, p.CategoryID, c.CategoryName
# MAGIC ),
# MAGIC ranked_products AS (
# MAGIC   SELECT 
# MAGIC     *,
# MAGIC     SUM(total_revenue) OVER () as grand_total_revenue,
# MAGIC     SUM(total_revenue) OVER (ORDER BY total_revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumulative_revenue,
# MAGIC     ROW_NUMBER() OVER (ORDER BY total_revenue DESC) as revenue_rank
# MAGIC   FROM product_revenue
# MAGIC )
# MAGIC SELECT 
# MAGIC   ProductID,
# MAGIC   ProductName,
# MAGIC   CategoryID,
# MAGIC   CategoryName,
# MAGIC   total_revenue,
# MAGIC   total_quantity_sold,
# MAGIC   transaction_count,
# MAGIC   revenue_rank,
# MAGIC   ROUND(total_revenue / grand_total_revenue * 100, 2) as revenue_percentage,
# MAGIC   ROUND(cumulative_revenue / grand_total_revenue * 100, 2) as cumulative_percentage,
# MAGIC   CASE 
# MAGIC     WHEN cumulative_revenue / grand_total_revenue <= 0.80 THEN 'A'
# MAGIC     WHEN cumulative_revenue / grand_total_revenue <= 0.95 THEN 'B'
# MAGIC     ELSE 'C'
# MAGIC   END as abc_category
# MAGIC FROM ranked_products
# MAGIC ORDER BY revenue_rank;
# MAGIC   
# MAGIC -- Display ABC Analysis results
# MAGIC SELECT * FROM abc_analysis LIMIT 50;

# COMMAND ----------

# DBTITLE 1,ABC Summary Statistics
# ABC Analysis Summary
abc_summary = spark.sql("""
  SELECT 
    abc_category,
    COUNT(*) as product_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as product_percentage,
    ROUND(SUM(total_revenue), 2) as total_revenue,
    ROUND(SUM(total_revenue) * 100.0 / SUM(SUM(total_revenue)) OVER (), 2) as revenue_percentage,
    ROUND(AVG(total_revenue), 2) as avg_revenue_per_product,
    MIN(total_revenue) as min_revenue,
    MAX(total_revenue) as max_revenue
  FROM abc_analysis
  GROUP BY abc_category
  ORDER BY abc_category
""")

display(abc_summary)

# COMMAND ----------

# DBTITLE 1,FSN Header
# MAGIC %md
# MAGIC ### 2.2 FSN Analysis (Movement-Based Classification)
# MAGIC
# MAGIC Classify products based on consumption/sales frequency:
# MAGIC - **Fast-moving**: High sales frequency (frequent purchases)
# MAGIC - **Slow-moving**: Low sales frequency (occasional purchases)
# MAGIC - **Non-moving**: No recent sales activity

# COMMAND ----------

# DBTITLE 1,FSN Analysis - Theory & Preparation
# MAGIC %md
# MAGIC ### 📚 FSN Analysis - Theory & Viva Preparation
# MAGIC
# MAGIC #### **Definition**
# MAGIC FSN Analysis classifies inventory based on **consumption rate and movement velocity** - how quickly items are consumed or sold.
# MAGIC
# MAGIC #### **Classification**
# MAGIC - **Fast-Moving (F)**: High consumption rate, frequent transactions, rapid turnover
# MAGIC - **Slow-Moving (S)**: Moderate consumption, occasional purchases, slower turnover
# MAGIC - **Non-Moving (N)**: Very low/no consumption, obsolete or dead stock
# MAGIC
# MAGIC #### **Key Metrics**
# MAGIC 1. **Consumption Rate**: Units consumed per day (primary indicator)
# MAGIC    - Formula: `Total Quantity / Days in Period`
# MAGIC 2. **Transaction Velocity**: Transactions per day
# MAGIC    - Formula: `Transaction Count / Days in Period`
# MAGIC 3. **Sales Frequency Ratio**: Percentage of days with sales
# MAGIC    - Formula: `Days with Sales / Total Days`
# MAGIC 4. **Days Since Last Sale**: Time since last transaction (staleness indicator)
# MAGIC
# MAGIC #### **Classification Methodology**
# MAGIC - **Fast-Moving**: Top 25-35% on consumption rate AND/OR high frequency
# MAGIC - **Non-Moving**: Bottom 30-40% on consumption OR >90 days without sales
# MAGIC - **Slow-Moving**: Middle tier (everything between F and N)
# MAGIC
# MAGIC Note: Unlike equal-distribution methods (NTILE), we use **percentile thresholds** for business-accurate, non-uniform distributions.
# MAGIC
# MAGIC #### **Business Application**
# MAGIC - **Fast-Moving**: 
# MAGIC   - High reorder frequency
# MAGIC   - Never allow stockouts (revenue loss)
# MAGIC   - Negotiate better supplier terms (bulk discounts)
# MAGIC   - Consider JIT (Just-In-Time) inventory
# MAGIC   
# MAGIC - **Slow-Moving**:
# MAGIC   - Review demand patterns for optimization
# MAGIC   - Consider promotional strategies to increase velocity
# MAGIC   - Adjust reorder points to avoid excess inventory
# MAGIC   
# MAGIC - **Non-Moving**:
# MAGIC   - Investigate reasons (obsolete, seasonal, overstocked)
# MAGIC   - Liquidation or clearance sales
# MAGIC   - Discontinue or make-to-order only
# MAGIC   - Free up warehouse space and capital
# MAGIC
# MAGIC #### **Advantages**
# MAGIC - Identifies obsolete/dead stock early
# MAGIC - Optimizes warehouse space utilization
# MAGIC - Prevents overstocking of slow movers
# MAGIC - Ensures adequate stock for fast movers
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,FSN Analysis Calculation
# MAGIC %sql
# MAGIC -- FSN Analysis: Movement-based classification (including all products, even those with no sales)
# MAGIC CREATE OR REPLACE TEMP VIEW fsn_analysis AS
# MAGIC WITH sales_frequency AS (
# MAGIC   SELECT 
# MAGIC     s.ProductID,
# MAGIC     COUNT(DISTINCT DATE(s.SalesDate)) as days_with_sales,
# MAGIC     COUNT(DISTINCT s.SalesID) as transaction_count,
# MAGIC     SUM(s.Quantity) as total_quantity,
# MAGIC     DATEDIFF(MAX(s.SalesDate), MIN(s.SalesDate)) + 1 as days_in_period,
# MAGIC     MAX(s.SalesDate) as last_sale_date
# MAGIC   FROM commercial.base.sales s
# MAGIC   GROUP BY s.ProductID
# MAGIC ),
# MAGIC all_products AS (
# MAGIC   SELECT 
# MAGIC     p.ProductID,
# MAGIC     p.ProductName,
# MAGIC     p.CategoryID,
# MAGIC     c.CategoryName,
# MAGIC     COALESCE(sf.days_with_sales, 0) as days_with_sales,
# MAGIC     COALESCE(sf.transaction_count, 0) as transaction_count,
# MAGIC     COALESCE(sf.total_quantity, 0) as total_quantity,
# MAGIC     COALESCE(sf.days_in_period, 0) as days_in_period,
# MAGIC     sf.last_sale_date,
# MAGIC     -- Calculate days_since_last_sale using dataset's max sales date
# MAGIC     CASE 
# MAGIC       WHEN sf.last_sale_date IS NOT NULL THEN DATEDIFF((SELECT MAX(SalesDate) FROM commercial.base.sales), sf.last_sale_date)
# MAGIC       ELSE NULL
# MAGIC     END as days_since_last_sale,
# MAGIC     CASE 
# MAGIC       WHEN sf.days_in_period > 0 THEN ROUND(sf.days_with_sales * 1.0 / sf.days_in_period, 4)
# MAGIC       ELSE 0 
# MAGIC     END as sales_frequency_ratio,
# MAGIC     -- Calculate transaction velocity (transactions per day)
# MAGIC     CASE 
# MAGIC       WHEN sf.days_in_period > 0 THEN ROUND(sf.transaction_count * 1.0 / sf.days_in_period, 2)
# MAGIC       ELSE 0
# MAGIC     END as transaction_velocity,
# MAGIC     -- CONSUMPTION RATE: Units consumed per day (actual demand rate)
# MAGIC     CASE 
# MAGIC       WHEN sf.days_in_period > 0 THEN ROUND(sf.total_quantity * 1.0 / sf.days_in_period, 2)
# MAGIC       ELSE 0
# MAGIC     END as consumption_rate,
# MAGIC     -- Average units per transaction (order size pattern)
# MAGIC     CASE 
# MAGIC       WHEN sf.transaction_count > 0 THEN ROUND(sf.total_quantity * 1.0 / sf.transaction_count, 2)
# MAGIC       ELSE 0
# MAGIC     END as avg_units_per_transaction
# MAGIC   FROM commercial.base.products p
# MAGIC   LEFT JOIN commercial.base.categories c ON p.CategoryID = c.CategoryID
# MAGIC   LEFT JOIN sales_frequency sf ON p.ProductID = sf.ProductID
# MAGIC )
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   -- Calculate percentile ranks for movement metrics
# MAGIC   PERCENT_RANK() OVER (ORDER BY consumption_rate DESC) as consumption_percentile,
# MAGIC   PERCENT_RANK() OVER (ORDER BY transaction_velocity DESC) as velocity_percentile,
# MAGIC   PERCENT_RANK() OVER (ORDER BY sales_frequency_ratio DESC) as frequency_percentile,
# MAGIC   -- Composite movement score: weighted combination of consumption rate and frequency
# MAGIC   (consumption_rate * 0.6) + (transaction_velocity * 100 * 0.3) + (sales_frequency_ratio * 100 * 0.1) as movement_score,
# MAGIC   -- FSN Classification based on consumption rate + frequency:
# MAGIC   -- Fast-moving: High consumption (top 25%) OR high consumption+frequency (top 35%)
# MAGIC   -- Non-moving: Very low consumption (bottom 30%) OR no recent activity
# MAGIC   -- Slow-moving: Everything in between
# MAGIC   CASE 
# MAGIC     -- Fast-Moving: High consumption rate (primary indicator)
# MAGIC     WHEN consumption_rate >= PERCENTILE(consumption_rate, 0.75) OVER () THEN 'Fast-Moving'
# MAGIC     -- Fast-Moving: Good consumption + high frequency
# MAGIC     WHEN consumption_rate >= PERCENTILE(consumption_rate, 0.65) OVER () 
# MAGIC          AND transaction_velocity >= PERCENTILE(transaction_velocity, 0.65) OVER () THEN 'Fast-Moving'
# MAGIC     -- Non-Moving: Very low consumption
# MAGIC     WHEN consumption_rate <= PERCENTILE(consumption_rate, 0.30) OVER () THEN 'Non-Moving'
# MAGIC     -- Non-Moving: No recent sales (stale inventory)
# MAGIC     WHEN days_since_last_sale > 90 OR days_since_last_sale IS NULL THEN 'Non-Moving'
# MAGIC     -- Non-Moving: Low consumption + low frequency
# MAGIC     WHEN consumption_rate <= PERCENTILE(consumption_rate, 0.40) OVER ()
# MAGIC          AND transaction_velocity <= PERCENTILE(transaction_velocity, 0.40) OVER () THEN 'Non-Moving'
# MAGIC     -- Slow-Moving: Everything else (moderate consumption/frequency)
# MAGIC     ELSE 'Slow-Moving'
# MAGIC   END as fsn_category
# MAGIC FROM all_products
# MAGIC ORDER BY consumption_rate DESC, transaction_velocity DESC;
# MAGIC
# MAGIC -- Display FSN results for all three categories
# MAGIC SELECT  * FROM fsn_analysis  LIMIT 50;

# COMMAND ----------

# DBTITLE 1,FSN Summary Statistics
# FSN Analysis Summary with Consumption Metrics
fsn_summary = spark.sql("""
  SELECT 
    fsn_category,
    COUNT(*) as product_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as product_percentage,
    ROUND(AVG(consumption_rate), 2) as avg_consumption_rate_units_per_day,
    ROUND(AVG(transaction_velocity), 2) as avg_transaction_velocity,
    ROUND(AVG(sales_frequency_ratio), 4) as avg_frequency_ratio,
    ROUND(AVG(days_since_last_sale), 1) as avg_days_since_sale,
    SUM(total_quantity) as total_units_sold,
    ROUND(AVG(transaction_count), 1) as avg_transactions,
    ROUND(AVG(avg_units_per_transaction), 1) as avg_order_size,
    ROUND(AVG(movement_score), 2) as avg_movement_score
  FROM fsn_analysis
  GROUP BY fsn_category
  ORDER BY 
    CASE fsn_category 
      WHEN 'Fast-Moving' THEN 1 
      WHEN 'Slow-Moving' THEN 2 
      ELSE 3 
    END
""")

display(fsn_summary)

# COMMAND ----------

# DBTITLE 1,VED Header
# MAGIC %md
# MAGIC ### 2.3 VED Analysis (Criticality-Based Classification)
# MAGIC
# MAGIC Classify products based on business criticality:
# MAGIC - **Vital**: Critical products (high velocity, frequent in cancelled orders due to stockouts)
# MAGIC - **Essential**: Moderately important
# MAGIC - **Desirable**: Nice to have, can tolerate stockouts

# COMMAND ----------

# DBTITLE 1,VED Analysis - Theory & Preparation
# MAGIC %md
# MAGIC ### 📚 VED Analysis - Theory & Viva Preparation
# MAGIC
# MAGIC #### **Definition**
# MAGIC VED Analysis classifies inventory based on **business criticality** - the operational impact of stockouts or unavailability.
# MAGIC
# MAGIC #### **Classification**
# MAGIC - **Vital (V)**: Critical items - stockout causes business disruption, lost sales, or customer dissatisfaction
# MAGIC - **Essential (E)**: Moderately important - short-term shortage tolerable but should be avoided
# MAGIC - **Desirable (D)**: Nice to have - stockouts cause minimal business impact, can be substituted
# MAGIC
# MAGIC #### **Criticality Indicators**
# MAGIC 1. **Daily Velocity**: High velocity items are often critical (frequent demand = high importance)
# MAGIC 2. **Stockout Impact**: Items frequently cancelled due to stockouts are vital
# MAGIC 3. **Customer Impact**: Items whose absence leads to customer complaints/lost sales
# MAGIC 4. **Substitutability**: Items without substitutes are more critical
# MAGIC 5. **Operational Dependency**: Items that halt production/operations when unavailable
# MAGIC
# MAGIC #### **Classification Methodology**
# MAGIC We use a **composite criticality score** combining:
# MAGIC - Daily velocity (70% weight) - operational importance
# MAGIC - Stockout cancellations (30% weight) - proven customer impact
# MAGIC
# MAGIC Thresholds:
# MAGIC - **Vital**: Top 25-30% velocity OR significant stockout history (>3 cancellations)
# MAGIC - **Desirable**: Bottom 40-45% velocity (low impact if unavailable)
# MAGIC - **Essential**: Middle tier (moderate importance)
# MAGIC
# MAGIC #### **Business Application**
# MAGIC - **Vital Items**:
# MAGIC   - Never allow stockouts - highest priority
# MAGIC   - Multiple suppliers for redundancy
# MAGIC   - Higher safety stock (3-4 weeks)
# MAGIC   - Real-time inventory monitoring
# MAGIC   - Expedited ordering when needed
# MAGIC   - May stock even if C-class (low value)
# MAGIC   
# MAGIC - **Essential Items**:
# MAGIC   - Maintain reasonable safety stock (1-2 weeks)
# MAGIC   - Regular monitoring
# MAGIC   - Standard reorder procedures
# MAGIC   
# MAGIC - **Desirable Items**:
# MAGIC   - Minimal safety stock or make-to-order
# MAGIC   - Extended lead times acceptable
# MAGIC   - Can consider stockouts temporarily
# MAGIC
# MAGIC #### **VED vs ABC - Key Difference**
# MAGIC A classic example: **Spare parts**
# MAGIC - A critical machine spare part may be:
# MAGIC   - **C-class** (low annual value - $100/year)
# MAGIC   - **Vital** (machine stops without it - $10,000/hour downtime)
# MAGIC   
# MAGIC ABC says "low priority," VED says "highest priority." **VED captures what ABC misses.**
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,VED Analysis Calculation
# MAGIC %sql
# MAGIC -- VED Analysis: Criticality-based classification
# MAGIC CREATE OR REPLACE TEMP VIEW ved_analysis AS
# MAGIC WITH product_criticality AS (
# MAGIC   SELECT 
# MAGIC     s.ProductID,
# MAGIC     p.ProductName,
# MAGIC     SUM(s.Quantity) as total_quantity_sold,
# MAGIC     COUNT(DISTINCT s.SalesID) as transaction_count,
# MAGIC     AVG(s.Quantity) as avg_order_quantity,
# MAGIC     STDDEV(s.Quantity) as stddev_quantity,
# MAGIC     -- Calculate sales velocity (units per day)
# MAGIC     SUM(s.Quantity) * 1.0 / NULLIF(DATEDIFF(MAX(s.SalesDate), MIN(s.SalesDate)) + 1, 0) as daily_velocity
# MAGIC   FROM commercial.base.sales s
# MAGIC   JOIN commercial.base.products p ON s.ProductID = p.ProductID
# MAGIC   GROUP BY s.ProductID, p.ProductName
# MAGIC ),
# MAGIC cancellation_impact AS (
# MAGIC   SELECT 
# MAGIC     ProductID,
# MAGIC     COUNT(*) as cancellation_count,
# MAGIC     SUM(CASE WHEN LOWER(CancellationReason) LIKE '%stock%' OR LOWER(CancellationReason) LIKE '%available%' THEN 1 ELSE 0 END) as stockout_cancellations
# MAGIC   FROM commercial.base.sales_cancelled_returned
# MAGIC   GROUP BY ProductID
# MAGIC ),
# MAGIC combined_criticality AS (
# MAGIC   SELECT 
# MAGIC     pc.*,
# MAGIC     COALESCE(ci.cancellation_count, 0) as cancellation_count,
# MAGIC     COALESCE(ci.stockout_cancellations, 0) as stockout_cancellations,
# MAGIC     -- Criticality score based on velocity and stockout impact
# MAGIC     COALESCE(pc.daily_velocity, 0) * 0.7 + COALESCE(ci.stockout_cancellations, 0) * 100 as criticality_score
# MAGIC   FROM product_criticality pc
# MAGIC   LEFT JOIN cancellation_impact ci ON pc.ProductID = ci.ProductID
# MAGIC )
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   -- VED Classification based on criticality (non-equal distribution):
# MAGIC   -- Vital: Top 25% daily velocity OR top 30% with stockout issues (critical for business)
# MAGIC   -- Desirable: Bottom 40% velocity (can tolerate stockouts, low business impact)
# MAGIC   -- Essential: Middle tier (moderate importance)
# MAGIC   CASE 
# MAGIC     -- Vital: High daily velocity (top 25%) - critical fast-movers
# MAGIC     WHEN daily_velocity >= PERCENTILE(daily_velocity, 0.75) OVER () THEN 'Vital'
# MAGIC     -- Vital: Good velocity (top 40%) with stockout impact
# MAGIC     WHEN daily_velocity >= PERCENTILE(daily_velocity, 0.60) OVER () 
# MAGIC          AND stockout_cancellations > 0 THEN 'Vital'
# MAGIC     -- Vital: High cancellation/stockout impact (indicates critical item)
# MAGIC     WHEN stockout_cancellations >= 3 THEN 'Vital'
# MAGIC     -- Desirable: Low velocity (bottom 40%) - can tolerate stockouts
# MAGIC     WHEN daily_velocity <= PERCENTILE(daily_velocity, 0.40) OVER () THEN 'Desirable'
# MAGIC     -- Desirable: Very low velocity (bottom 45%) even without stockouts
# MAGIC     WHEN daily_velocity <= PERCENTILE(daily_velocity, 0.45) OVER () 
# MAGIC          AND stockout_cancellations = 0 THEN 'Desirable'
# MAGIC     -- Essential: Everything else (moderate velocity and importance)
# MAGIC     ELSE 'Essential'
# MAGIC   END as ved_category
# MAGIC FROM combined_criticality
# MAGIC ORDER BY criticality_score DESC;
# MAGIC
# MAGIC -- Display VED results
# MAGIC SELECT * FROM ved_analysis LIMIT 50;

# COMMAND ----------

# DBTITLE 1,VED Summary Statistics
# VED Analysis Summary
ved_summary = spark.sql("""
  SELECT 
    ved_category,
    COUNT(*) as product_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as product_percentage,
    ROUND(AVG(daily_velocity), 2) as avg_daily_velocity,
    ROUND(AVG(criticality_score), 2) as avg_criticality_score,
    SUM(stockout_cancellations) as total_stockout_cancellations,
    SUM(total_quantity_sold) as total_units_sold
  FROM ved_analysis
  GROUP BY ved_category
  ORDER BY 
    CASE ved_category 
      WHEN 'Vital' THEN 1 
      WHEN 'Essential' THEN 2 
      ELSE 3 
    END
""")

display(ved_summary)

# COMMAND ----------

# DBTITLE 1,HML Header
# MAGIC %md
# MAGIC ### 2.4 HML Analysis (Price-Based Classification)
# MAGIC
# MAGIC Classify products based on unit price:
# MAGIC - **High**: Premium/expensive products
# MAGIC - **Medium**: Mid-range products  
# MAGIC - **Low**: Budget/economy products

# COMMAND ----------

# DBTITLE 1,HML Analysis - Theory & Preparation
# MAGIC %md
# MAGIC ### HML Analysis - Theory & Viva Preparation
# MAGIC
# MAGIC #### **Definition**
# MAGIC HML Analysis classifies inventory based on **unit price** - the cost per item. This helps in managing high-value inventory investments and theft prevention.
# MAGIC
# MAGIC #### **Classification**
# MAGIC - **High (H)**: Premium/expensive products (top 30% by price)
# MAGIC - **Medium (M)**: Mid-range products (middle tier, ~35%)
# MAGIC - **Low (L)**: Budget/economy products (bottom 35% by price)
# MAGIC
# MAGIC #### **Price-Based Inventory Implications**
# MAGIC
# MAGIC **High-Priced Items**:
# MAGIC - Higher holding costs (more capital tied up)
# MAGIC - Greater theft/damage risk
# MAGIC - Requires secure storage
# MAGIC - More accurate demand forecasting needed
# MAGIC - Economic Order Quantity (EOQ) critical
# MAGIC - Insurance considerations
# MAGIC
# MAGIC **Medium-Priced Items**:
# MAGIC - Standard inventory procedures
# MAGIC - Balanced approach to ordering
# MAGIC - Regular security measures
# MAGIC
# MAGIC **Low-Priced Items**:
# MAGIC - Lower holding cost per unit
# MAGIC - Can order in bulk (lower per-unit cost)
# MAGIC - Less security concern
# MAGIC - Simplified tracking acceptable
# MAGIC
# MAGIC #### **Business Application**
# MAGIC - **High-Price Items**:
# MAGIC   - Smaller, frequent orders (reduce capital tied up)
# MAGIC   - Enhanced security and tracking (prevent theft/loss)
# MAGIC   - Insurance for high-value inventory
# MAGIC   - Accurate demand forecasting (avoid overstock)
# MAGIC   - RFID or serial number tracking
# MAGIC   
# MAGIC - **Low-Price Items**:
# MAGIC   - Bulk ordering (minimize ordering cost)
# MAGIC   - Simpler tracking systems
# MAGIC   - Standard storage conditions
# MAGIC   - Higher tolerance for inventory variance
# MAGIC
# MAGIC #### **HML in Financial Context**
# MAGIC **Working Capital Impact**:
# MAGIC - High-price items tie up significant working capital
# MAGIC - Formula: `Inventory Value = Quantity × Unit Price`
# MAGIC - Optimizing high-price inventory frees cash for other investments
# MAGIC
# MAGIC **Carrying Cost Considerations**:
# MAGIC - Carrying cost as % of inventory value (typically 20-30% annually)
# MAGIC - High-price items have proportionally higher carrying costs
# MAGIC - Includes: storage, insurance, obsolescence, opportunity cost
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,HML Analysis Calculation
# MAGIC %sql
# MAGIC -- HML Analysis: Price-based classification (non-equal distribution)
# MAGIC CREATE OR REPLACE TEMP VIEW hml_analysis AS
# MAGIC SELECT 
# MAGIC   ProductID,
# MAGIC   ProductName,
# MAGIC   Price,
# MAGIC   CategoryID,
# MAGIC   -- HML Classification based on unit price percentiles:
# MAGIC   -- High: Top 30% of prices (premium/expensive products)
# MAGIC   -- Low: Bottom 35% of prices (budget/economy products)
# MAGIC   -- Medium: Middle tier (mid-range products)
# MAGIC   CASE 
# MAGIC     WHEN Price >= PERCENTILE(Price, 0.70) OVER () THEN 'High'
# MAGIC     WHEN Price <= PERCENTILE(Price, 0.35) OVER () THEN 'Low'
# MAGIC     ELSE 'Medium'
# MAGIC   END as hml_category
# MAGIC FROM commercial.base.products
# MAGIC WHERE Price IS NOT NULL
# MAGIC ORDER BY Price DESC;
# MAGIC
# MAGIC -- Display HML results
# MAGIC SELECT * FROM hml_analysis LIMIT 50;

# COMMAND ----------

# DBTITLE 1,HML Summary Statistics
# HML Analysis Summary
hml_summary = spark.sql("""
  SELECT 
    hml_category,
    COUNT(*) as product_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as product_percentage,
    ROUND(MIN(Price), 2) as min_price,
    ROUND(MAX(Price), 2) as max_price,
    ROUND(AVG(Price), 2) as avg_price,
    ROUND(PERCENTILE(Price, 0.5), 2) as median_price
  FROM hml_analysis
  GROUP BY hml_category
  ORDER BY 
    CASE hml_category 
      WHEN 'High' THEN 1 
      WHEN 'Medium' THEN 2 
      ELSE 3 
    END
""")

display(hml_summary)

# COMMAND ----------

# DBTITLE 1,Combined Matrix Header
# MAGIC %md
# MAGIC ### 2.5 Combined ABC-FSN-VED-HML Matrix
# MAGIC
# MAGIC Integrated inventory classification combining four key dimensions:
# MAGIC - **ABC**: Value/revenue contribution (A=high, B=medium, C=low)
# MAGIC - **FSN**: Movement/consumption rate (Fast/Slow/Non-moving)
# MAGIC - **VED**: Business criticality (Vital/Essential/Desirable)
# MAGIC - **HML**: Unit price (High/Medium/Low)
# MAGIC
# MAGIC This multi-dimensional analysis provides comprehensive inventory policy recommendations.

# COMMAND ----------

# DBTITLE 1,Combined Matrix - Theory & Preparation
# MAGIC %md
# MAGIC ###  Combined ABC-FSN-VED-HML Matrix - Theory & Viva Preparation
# MAGIC
# MAGIC #### **Definition**
# MAGIC Integrated **multi-dimensional inventory classification** combining four complementary perspectives to create comprehensive, nuanced inventory policies.
# MAGIC
# MAGIC #### **Why Multi-Dimensional Analysis?**
# MAGIC Single-dimension analysis is insufficient:
# MAGIC - **ABC alone**: Ignores movement, criticality, unit price
# MAGIC - **FSN alone**: Ignores value, criticality
# MAGIC - **VED alone**: Ignores movement speed, value
# MAGIC - **HML alone**: Ignores total value, movement, criticality
# MAGIC
# MAGIC **Real-world inventory requires ALL four dimensions for optimal decisions.**
# MAGIC
# MAGIC #### **The Four Dimensions**
# MAGIC
# MAGIC | Dimension | Perspective | Question Answered |
# MAGIC |-----------|-------------|-------------------|
# MAGIC | **ABC** | Financial | "How much revenue does it generate?" |
# MAGIC | **FSN** | Operational | "How fast does it move?" |
# MAGIC | **VED** | Strategic | "How critical is it to operations?" |
# MAGIC | **HML** | Investment | "How much capital per unit?" |
# MAGIC
# MAGIC #### **Combined Classification Examples**
# MAGIC
# MAGIC **1. A-Fast-Vital-High** (❗ CRITICAL)
# MAGIC - High revenue, fast-moving, business-critical, expensive
# MAGIC - **Example**: Flagship smartphone in electronics retail
# MAGIC - **Policy**: Hourly monitoring, 3-4 weeks safety stock, automated reorder, multiple suppliers, never stock out
# MAGIC
# MAGIC **2. A-Fast-Essential-Low** (🔺 HIGH PRIORITY)
# MAGIC - High revenue, fast-moving, moderately important, cheap
# MAGIC - **Example**: Soft drinks in convenience store
# MAGIC - **Policy**: Daily monitoring, 1-2 weeks safety stock, bulk ordering (low per-unit cost)
# MAGIC
# MAGIC **3. C-Non-Moving-Desirable-Low** (❌ DISCONTINUE)
# MAGIC - Low revenue, no movement, not critical, cheap
# MAGIC - **Example**: Obsolete accessories, out-of-season items
# MAGIC - **Policy**: Clearance sale, discontinue, free up space
# MAGIC
# MAGIC **4. C-Slow-Vital-High** (⚠️ SPECIAL CASE)
# MAGIC - Low total revenue, slow-moving, critical, expensive per unit
# MAGIC - **Example**: Critical spare parts for machinery
# MAGIC - **Policy**: Maintain stock despite low revenue (criticality overrides value), but minimize quantity to reduce capital tied up
# MAGIC
# MAGIC #### **Decision Hierarchy**
# MAGIC **When dimensions conflict, priority order**:
# MAGIC 1. **VED (Criticality)** - Never compromise on Vital items
# MAGIC 2. **FSN (Movement)** - Fast-movers need availability
# MAGIC 3. **ABC (Value)** - Revenue impact
# MAGIC 4. **HML (Price)** - Unit-level investment considerations
# MAGIC
# MAGIC **Example of conflict resolution**:
# MAGIC - Item is **C-Fast-Vital-Low**
# MAGIC - ABC says "low priority" (C-class)
# MAGIC - VED says "highest priority" (Vital)
# MAGIC - FSN says "prevent stockout" (Fast)
# MAGIC - **Decision**: Stock adequately (VED+FSN override ABC)
# MAGIC
# MAGIC #### **Inventory Policy Matrix**
# MAGIC
# MAGIC | Policy Level | Example Combinations | Action |
# MAGIC |--------------|---------------------|--------|
# MAGIC | **CRITICAL** | A-Fast-Vital, A-Fast-Vital-High | Hourly monitoring, never stock out, 3-4 weeks safety stock |
# MAGIC | **HIGH PRIORITY** | A-Fast, A-Vital, Fast-Vital | Daily/weekly review, 1-2 weeks safety stock, automated reorder |
# MAGIC | **OPTIMIZE** | Slow-moving combinations | Review demand patterns, adjust reorder points, consider promotions |
# MAGIC | **LOW PRIORITY** | Non-moving or Desirable+Low | Minimal stock, extended lead times OK, monthly review |
# MAGIC | **DISCONTINUE** | C-Non-Desirable-Low | Clearance, phase-out, free up capital and space |
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Combined Classification Matrix
# MAGIC %sql
# MAGIC -- Combined ABC-FSN-VED-HML Classification Matrix
# MAGIC CREATE OR REPLACE TEMP VIEW combined_inventory_classification AS
# MAGIC SELECT 
# MAGIC   a.ProductID,
# MAGIC   a.ProductName,
# MAGIC   a.CategoryName,
# MAGIC   a.abc_category,
# MAGIC   f.fsn_category,
# MAGIC   v.ved_category,
# MAGIC   h.hml_category,
# MAGIC   CONCAT(a.abc_category, '-', f.fsn_category, '-', v.ved_category, '-', h.hml_category) as inventory_class,
# MAGIC   a.total_revenue,
# MAGIC   a.cumulative_percentage as abc_cumulative_pct,
# MAGIC   f.consumption_rate,
# MAGIC   f.transaction_velocity,
# MAGIC   f.days_since_last_sale,
# MAGIC   v.daily_velocity,
# MAGIC   v.criticality_score,
# MAGIC   v.stockout_cancellations,
# MAGIC   h.Price as unit_price,
# MAGIC   -- Recommended inventory policy (considering ABC-FSN-VED-HML)
# MAGIC   CASE 
# MAGIC     -- CRITICAL: A-Fast-Vital-High (highest value, high movement, critical, premium)
# MAGIC     WHEN a.abc_category = 'A' AND f.fsn_category = 'Fast-Moving' AND v.ved_category = 'Vital' AND h.hml_category = 'High'
# MAGIC       THEN 'CRITICAL: Hourly monitoring, high safety stock (3-4 weeks), automated reorder, expedite suppliers'
# MAGIC     -- CRITICAL: A-Fast-Vital (high value, high movement, critical)
# MAGIC     WHEN a.abc_category = 'A' AND f.fsn_category = 'Fast-Moving' AND v.ved_category = 'Vital'
# MAGIC       THEN 'CRITICAL: Daily monitoring, high safety stock (2-3 weeks), automated reorder'
# MAGIC     -- HIGH PRIORITY: A-Fast or A-Vital (high value with either fast movement or critical)
# MAGIC     WHEN a.abc_category = 'A' AND (f.fsn_category = 'Fast-Moving' OR v.ved_category = 'Vital')
# MAGIC       THEN 'HIGH PRIORITY: Daily/weekly monitoring, moderate safety stock (1-2 weeks)'
# MAGIC     -- ATTENTION: Fast-moving items regardless of value (prevent stockouts)
# MAGIC     WHEN f.fsn_category = 'Fast-Moving' AND v.ved_category = 'Vital'
# MAGIC       THEN 'HIGH PRIORITY: Prevent stockouts, monitor weekly, safety stock (1-2 weeks)'
# MAGIC     -- REVIEW: C-Non-Moving-Desirable-Low (low value, no movement, not critical, cheap)
# MAGIC     WHEN a.abc_category = 'C' AND f.fsn_category = 'Non-Moving' AND v.ved_category = 'Desirable' AND h.hml_category = 'Low'
# MAGIC       THEN 'DISCONTINUE: Consider removal, make-to-order only, or clearance sale'
# MAGIC     -- LOW PRIORITY: Non-moving or desirable + low price
# MAGIC     WHEN f.fsn_category = 'Non-Moving' OR (v.ved_category = 'Desirable' AND h.hml_category = 'Low')
# MAGIC       THEN 'LOW PRIORITY: Minimal stock, extended lead times acceptable, monthly review'
# MAGIC     -- SLOW MOVERS: Require optimization
# MAGIC     WHEN f.fsn_category = 'Slow-Moving'
# MAGIC       THEN 'OPTIMIZE: Review demand patterns, adjust reorder points, consider promotions'
# MAGIC     -- STANDARD: Everything else
# MAGIC     ELSE 'STANDARD: Normal monitoring and reorder protocols, safety stock (1 week)'
# MAGIC   END as inventory_policy
# MAGIC FROM abc_analysis a
# MAGIC JOIN fsn_analysis f ON a.ProductID = f.ProductID
# MAGIC JOIN ved_analysis v ON a.ProductID = v.ProductID
# MAGIC JOIN hml_analysis h ON a.ProductID = h.ProductID
# MAGIC ORDER BY a.total_revenue DESC;
# MAGIC
# MAGIC -- Display combined classification
# MAGIC SELECT * FROM combined_inventory_classification LIMIT 100;

# COMMAND ----------

# DBTITLE 1,Combined Matrix Summary
# Combined Classification Summary
combined_summary = spark.sql("""
  SELECT 
    inventory_class,
    COUNT(*) as product_count,
    inventory_policy,
    ROUND(SUM(total_revenue), 2) as total_revenue,
    ROUND(AVG(daily_velocity), 2) as avg_velocity,
    SUM(stockout_cancellations) as stockout_count
  FROM combined_inventory_classification
  GROUP BY inventory_class, inventory_policy
  ORDER BY total_revenue DESC
  LIMIT 20
""")

display(combined_summary)

# Policy distribution
policy_summary = spark.sql("""
  SELECT 
    inventory_policy,
    COUNT(*) as product_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    ROUND(SUM(total_revenue), 2) as total_revenue
  FROM combined_inventory_classification
  GROUP BY inventory_policy
  ORDER BY total_revenue DESC
""")

display(policy_summary)

# COMMAND ----------

# DBTITLE 1,Objective 2 Header
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 3. Objective 2: Customer Segmentation Analysis
# MAGIC
# MAGIC ### 3.1 RFM Analysis (Recency, Frequency, Monetary)
# MAGIC
# MAGIC Segment customers based on purchasing behavior to identify high-value customers and at-risk segments.

# COMMAND ----------

# DBTITLE 1,RFM Analysis - Theory & Preparation
# MAGIC %md
# MAGIC ### 📚 RFM Analysis - Theory & Viva Preparation
# MAGIC
# MAGIC #### **Definition**
# MAGIC RFM Analysis is a **customer segmentation technique** based on three behavioral dimensions:
# MAGIC - **Recency (R)**: How recently did the customer purchase?
# MAGIC - **Frequency (F)**: How often do they purchase?
# MAGIC - **Monetary (M)**: How much do they spend?
# MAGIC
# MAGIC #### **The RFM Model - Core Concept**
# MAGIC
# MAGIC **Why these three dimensions?**
# MAGIC 1. **Recency**: Recent buyers are more likely to buy again (customer engagement)
# MAGIC 2. **Frequency**: Frequent buyers show loyalty and habit formation
# MAGIC 3. **Monetary**: High spenders contribute more revenue and profit
# MAGIC
# MAGIC **Predictive Power**: RFM strongly predicts future purchase behavior and customer lifetime value (CLV).
# MAGIC
# MAGIC #### **RFM Scoring Methodology**
# MAGIC
# MAGIC **Step 1: Calculate Metrics**
# MAGIC - **Recency**: Days since last purchase (from reference date)
# MAGIC   - `Recency = Reference_Date - Last_Purchase_Date`
# MAGIC   - **Lower is better** (recent purchase)
# MAGIC   
# MAGIC - **Frequency**: Number of transactions in analysis period
# MAGIC   - `Frequency = COUNT(DISTINCT Transactions)`
# MAGIC   - **Higher is better** (more purchases)
# MAGIC   
# MAGIC - **Monetary**: Total spend in analysis period
# MAGIC   - `Monetary = SUM(Total_Price)`
# MAGIC   - **Higher is better** (more revenue)
# MAGIC
# MAGIC **Step 2: Score Each Dimension (1-5 scale)**
# MAGIC - Divide customers into **quintiles** (5 equal groups) for each dimension
# MAGIC - **R-score**: 5 = most recent, 1 = least recent
# MAGIC - **F-score**: 5 = highest frequency, 1 = lowest frequency
# MAGIC - **M-score**: 5 = highest spend, 1 = lowest spend
# MAGIC
# MAGIC **Common Method**: Use `NTILE(5)` or percentiles (20%, 40%, 60%, 80%, 100%)
# MAGIC
# MAGIC **Step 3: Combine Scores**
# MAGIC - **RFM Score**: Concatenated string (e.g., "555", "432", "111")
# MAGIC - **RFM Total**: Sum of scores (e.g., 555 → 15, 111 → 3)
# MAGIC
# MAGIC #### **Customer Segments**
# MAGIC
# MAGIC | Segment | R | F | M | Description | Marketing Action |
# MAGIC |---------|---|---|---|-------------|------------------|
# MAGIC | **Champions** | 5 | 5 | 5 | Best customers - recent, frequent, high-spend | Reward programs, early access, VIP treatment |
# MAGIC | **Loyal Customers** | 4-5 | 4-5 | 4-5 | Regular, reliable customers | Upsell, loyalty programs, exclusive offers |
# MAGIC | **Potential Loyalists** | 4-5 | 1-2 | 1-2 | Recent buyers but low frequency/spend | Engagement campaigns, onboarding, education |
# MAGIC | **New Customers** | 5 | 1 | 1 | Just made first purchase | Welcome series, product recommendations |
# MAGIC | **At Risk** | 1-2 | 4-5 | 4-5 | Were loyal but haven't purchased recently | Win-back campaigns, special discounts |
# MAGIC | **Can't Lose Them** | 1-2 | 4-5 | 4-5 | High-value customers becoming inactive | Personalized retention, feedback requests |
# MAGIC | **Lost** | 1 | 1 | 1 | Inactive, low engagement | Reactivation campaigns or remove from lists |
# MAGIC | **Big Spenders** | any | any | 5 | High monetary value regardless of recency/frequency | Maintain relationship, prevent churn |
# MAGIC | **Others** | Mid | Mid | Mid | Average on all dimensions | Standard marketing, segment further |
# MAGIC
# MAGIC #### **Business Applications**
# MAGIC
# MAGIC **1. Marketing Personalization**
# MAGIC - **Champions**: "Thank you for being a valued customer! Here's exclusive early access..."
# MAGIC - **At Risk**: "We miss you! Here's 20% off your next purchase..."
# MAGIC - **New Customers**: "Welcome! Here's how to get the most out of your purchase..."
# MAGIC
# MAGIC **2. Resource Allocation**
# MAGIC - Focus retention efforts on "At Risk" and "Can't Lose Them" (high ROI)
# MAGIC - Invest in "Potential Loyalists" (conversion opportunity)
# MAGIC - Minimize spend on "Lost" customers (low probability of return)
# MAGIC
# MAGIC **3. Customer Lifetime Value (CLV) Prediction**
# MAGIC - Champions/Loyal have highest CLV → prioritize retention
# MAGIC - Potential Loyalists have growth potential → nurture relationship
# MAGIC - Lost customers have minimal future value → deprioritize
# MAGIC
# MAGIC **4. Channel Optimization**
# MAGIC - Champions: Premium channels (account manager, VIP support)
# MAGIC - At Risk: Targeted campaigns (email, retargeting ads)
# MAGIC - Lost: Minimal investment (automated reactivation only)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,RFM Analysis Calculation
# MAGIC %sql
# MAGIC -- RFM Analysis: Customer Segmentation
# MAGIC CREATE OR REPLACE TEMP VIEW rfm_analysis AS
# MAGIC WITH customer_metrics AS (
# MAGIC   SELECT 
# MAGIC     s.CustomerID,
# MAGIC     c.FirstName,
# MAGIC     c.LastName,
# MAGIC     c.CityID,
# MAGIC     -- Recency: Days since last purchase
# MAGIC     DATEDIFF(DATE('2018-05-31'), MAX(s.SalesDate)) as recency_days,
# MAGIC     -- Frequency: Number of transactions
# MAGIC     COUNT(DISTINCT s.SalesID) as frequency,
# MAGIC     -- Monetary: Total revenue
# MAGIC     SUM(s.TotalPrice) as monetary_value,
# MAGIC     -- Additional metrics
# MAGIC     COUNT(DISTINCT s.ProductID) as unique_products_purchased,
# MAGIC     AVG(s.TotalPrice) as avg_order_value,
# MAGIC     MIN(s.SalesDate) as first_purchase_date,
# MAGIC     MAX(s.SalesDate) as last_purchase_date,
# MAGIC     DATEDIFF(MAX(s.SalesDate), MIN(s.SalesDate)) as customer_lifespan_days
# MAGIC   FROM commercial.base.sales s
# MAGIC   JOIN commercial.base.customers c ON s.CustomerID = c.CustomerID
# MAGIC   GROUP BY s.CustomerID, c.FirstName, c.LastName, c.CityID
# MAGIC ),
# MAGIC rfm_scores AS (
# MAGIC   SELECT 
# MAGIC     *,
# MAGIC     -- RFM Scoring (1-5, where 5 is best)
# MAGIC     NTILE(5) OVER (ORDER BY recency_days ASC) as r_score,  -- Lower recency is better
# MAGIC     NTILE(5) OVER (ORDER BY frequency DESC) as f_score,
# MAGIC     NTILE(5) OVER (ORDER BY monetary_value DESC) as m_score,
# MAGIC     NTILE(5) OVER (ORDER BY unique_products_purchased DESC) as diversity_score
# MAGIC   FROM customer_metrics
# MAGIC )
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   CONCAT(r_score, f_score, m_score) as rfm_score,
# MAGIC   (r_score + f_score + m_score) as rfm_total_score,
# MAGIC   -- Customer Segmentation
# MAGIC   CASE 
# MAGIC     WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
# MAGIC     WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
# MAGIC     WHEN r_score >= 4 AND f_score <= 2 AND m_score <= 2 THEN 'New Customers'
# MAGIC     WHEN r_score >= 3 AND f_score <= 2 THEN 'Potential Loyalists'
# MAGIC     WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk'
# MAGIC     WHEN r_score <= 2 AND f_score >= 4 THEN 'Cant Lose Them'
# MAGIC     WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2 THEN 'Lost'
# MAGIC     WHEN m_score >= 4 THEN 'Big Spenders'
# MAGIC     ELSE 'Others'
# MAGIC   END as customer_segment
# MAGIC FROM rfm_scores
# MAGIC ORDER BY rfm_total_score DESC;
# MAGIC
# MAGIC -- Display RFM results
# MAGIC SELECT * FROM rfm_analysis LIMIT 50;

# COMMAND ----------

# DBTITLE 1,RFM Segment Summary
# Customer Segment Distribution
segment_summary = spark.sql("""
  SELECT 
    customer_segment,
    COUNT(*) as customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as customer_percentage,
    ROUND(AVG(monetary_value), 2) as avg_lifetime_value,
    ROUND(SUM(monetary_value), 2) as total_revenue,
    ROUND(SUM(monetary_value) * 100.0 / SUM(SUM(monetary_value)) OVER (), 2) as revenue_percentage,
    ROUND(AVG(frequency), 1) as avg_purchase_frequency,
    ROUND(AVG(recency_days), 1) as avg_recency_days,
    ROUND(AVG(unique_products_purchased), 1) as avg_product_diversity
  FROM rfm_analysis
  GROUP BY customer_segment
  ORDER BY total_revenue DESC
""")

display(segment_summary)

# COMMAND ----------

# DBTITLE 1,Discount Sensitivity Header
# MAGIC %md
# MAGIC ### 3.2 Customer Discount Sensitivity Analysis
# MAGIC
# MAGIC Analyze how different customer segments respond to discounts.

# COMMAND ----------

# DBTITLE 1,Discount Sensitivity Analysis
# MAGIC %sql
# MAGIC -- Customer Discount Sensitivity
# MAGIC CREATE OR REPLACE TEMP VIEW discount_sensitivity AS
# MAGIC SELECT 
# MAGIC   r.customer_segment,
# MAGIC   COUNT(DISTINCT s.SalesID) as total_transactions,
# MAGIC   SUM(CASE WHEN s.Discount > 0 THEN 1 ELSE 0 END) as discounted_transactions,
# MAGIC   ROUND(SUM(CASE WHEN s.Discount > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as discount_rate_pct,
# MAGIC   ROUND(AVG(s.Discount), 2) as avg_discount,
# MAGIC   ROUND(AVG(CASE WHEN s.Discount > 0 THEN s.Discount END), 2) as avg_discount_when_applied,
# MAGIC   ROUND(AVG(s.TotalPrice), 2) as avg_order_value,
# MAGIC   ROUND(AVG(CASE WHEN s.Discount > 0 THEN s.TotalPrice END), 2) as avg_order_value_with_discount,
# MAGIC   ROUND(AVG(CASE WHEN s.Discount = 0 THEN s.TotalPrice END), 2) as avg_order_value_without_discount
# MAGIC FROM commercial.base.sales s
# MAGIC JOIN rfm_analysis r ON s.CustomerID = r.CustomerID
# MAGIC GROUP BY r.customer_segment
# MAGIC ORDER BY total_transactions DESC;
# MAGIC
# MAGIC SELECT * FROM discount_sensitivity;

# COMMAND ----------

# DBTITLE 1,Objective 3 Header
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 4. Objective 3: Return & Cancellation Analysis
# MAGIC
# MAGIC ### 4.1 Return/Cancellation Metrics by Product and Category

# COMMAND ----------

# DBTITLE 1,Return Analysis by Product
# MAGIC %sql
# MAGIC -- Return and Cancellation Analysis
# MAGIC CREATE OR REPLACE TEMP VIEW return_analysis AS
# MAGIC WITH sales_summary AS (
# MAGIC   SELECT 
# MAGIC     ProductID,
# MAGIC     COUNT(*) as total_sales,
# MAGIC     SUM(Quantity) as total_quantity_sold,
# MAGIC     SUM(TotalPrice) as total_revenue
# MAGIC   FROM commercial.base.sales
# MAGIC   GROUP BY ProductID
# MAGIC ),
# MAGIC return_summary AS (
# MAGIC   SELECT 
# MAGIC     ProductID,
# MAGIC     COUNT(*) as return_count,
# MAGIC     SUM(Quantity) as returned_quantity,
# MAGIC     SUM(RefundAmount) as total_refund_amount,
# MAGIC     SUM(CASE WHEN OrderStatus = 'CANCELLED' THEN 1 ELSE 0 END) as cancelled_count,
# MAGIC     SUM(CASE WHEN OrderStatus = 'RETURNED' THEN 1 ELSE 0 END) as returned_count,
# MAGIC     -- Categorize reasons
# MAGIC     SUM(CASE WHEN LOWER(CancellationReason) LIKE '%defect%' OR LOWER(CancellationReason) LIKE '%quality%' THEN 1 ELSE 0 END) as quality_issues,
# MAGIC     SUM(CASE WHEN LOWER(CancellationReason) LIKE '%stock%' OR LOWER(CancellationReason) LIKE '%available%' THEN 1 ELSE 0 END) as stockout_issues,
# MAGIC     SUM(CASE WHEN LOWER(CancellationReason) LIKE '%price%' OR LOWER(CancellationReason) LIKE '%expensive%' THEN 1 ELSE 0 END) as price_issues,
# MAGIC     SUM(CASE WHEN LOWER(CancellationReason) LIKE '%delivery%' OR LOWER(CancellationReason) LIKE '%shipping%' THEN 1 ELSE 0 END) as delivery_issues
# MAGIC   FROM commercial.base.sales_cancelled_returned
# MAGIC   GROUP BY ProductID
# MAGIC )
# MAGIC SELECT 
# MAGIC   p.ProductID,
# MAGIC   p.ProductName,
# MAGIC   c.CategoryName,
# MAGIC   COALESCE(ss.total_sales, 0) as total_sales,
# MAGIC   COALESCE(rs.return_count, 0) as return_count,
# MAGIC   COALESCE(ss.total_quantity_sold, 0) as total_quantity_sold,
# MAGIC   COALESCE(rs.returned_quantity, 0) as returned_quantity,
# MAGIC   ROUND(COALESCE(rs.return_count, 0) * 100.0 / NULLIF(ss.total_sales, 0), 2) as return_rate_pct,
# MAGIC   ROUND(COALESCE(rs.returned_quantity, 0) * 100.0 / NULLIF(ss.total_quantity_sold, 0), 2) as quantity_return_rate_pct,
# MAGIC   ROUND(COALESCE(rs.total_refund_amount, 0), 2) as total_refunds,
# MAGIC   COALESCE(rs.cancelled_count, 0) as cancelled_count,
# MAGIC   COALESCE(rs.returned_count, 0) as returned_count,
# MAGIC   COALESCE(rs.quality_issues, 0) as quality_issues,
# MAGIC   COALESCE(rs.stockout_issues, 0) as stockout_issues,
# MAGIC   COALESCE(rs.price_issues, 0) as price_issues,
# MAGIC   COALESCE(rs.delivery_issues, 0) as delivery_issues,
# MAGIC   -- Product quality score (inverse of return rate)
# MAGIC   ROUND(100 - COALESCE(rs.return_count, 0) * 100.0 / NULLIF(ss.total_sales, 0), 2) as quality_score
# MAGIC FROM commercial.base.products p
# MAGIC LEFT JOIN commercial.base.categories c ON p.CategoryID = c.CategoryID
# MAGIC LEFT JOIN sales_summary ss ON p.ProductID = ss.ProductID
# MAGIC LEFT JOIN return_summary rs ON p.ProductID = rs.ProductID
# MAGIC WHERE ss.total_sales > 0
# MAGIC ORDER BY return_rate_pct DESC;
# MAGIC
# MAGIC SELECT * FROM return_analysis LIMIT 50;

# COMMAND ----------

# DBTITLE 1,Return Analysis Summary
# Return Analysis Summary by Category
category_return_summary = spark.sql("""
  SELECT 
    CategoryName,
    COUNT(*) as product_count,
    SUM(total_sales) as total_transactions,
    SUM(return_count) as total_returns,
    ROUND(AVG(return_rate_pct), 2) as avg_return_rate_pct,
    ROUND(SUM(total_refunds), 2) as total_refund_amount,
    SUM(quality_issues) as quality_issues,
    SUM(stockout_issues) as stockout_issues,
    SUM(price_issues) as price_issues,
    SUM(delivery_issues) as delivery_issues
  FROM return_analysis
  WHERE CategoryName IS NOT NULL
  GROUP BY CategoryName
  ORDER BY total_returns DESC
""")

display(category_return_summary)

# Top reasons for cancellations
reason_summary = spark.sql("""
  SELECT 
    CancellationReason,
    OrderStatus,
    COUNT(*) as occurrence_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    ROUND(SUM(RefundAmount), 2) as total_refund_amount
  FROM commercial.base.sales_cancelled_returned
  WHERE CancellationReason IS NOT NULL
  GROUP BY CancellationReason, OrderStatus
  ORDER BY occurrence_count DESC
  LIMIT 20
""")

display(reason_summary)

# COMMAND ----------

# DBTITLE 1,Objective 4 Header
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 5. Objective 4: Product Portfolio Analysis
# MAGIC
# MAGIC ### 5.1 Sales Velocity and Profitability Metrics

# COMMAND ----------

# DBTITLE 1,Product Portfolio Analysis
# MAGIC %sql
# MAGIC -- Product Portfolio Performance Analysis
# MAGIC CREATE OR REPLACE TEMP VIEW product_portfolio AS
# MAGIC WITH product_performance AS (
# MAGIC   SELECT 
# MAGIC     s.ProductID,
# MAGIC     p.ProductName,
# MAGIC     c.CategoryName,
# MAGIC     p.Price as unit_price,
# MAGIC     COUNT(DISTINCT s.SalesID) as transaction_count,
# MAGIC     SUM(s.Quantity) as total_units_sold,
# MAGIC     SUM(s.TotalPrice) as total_revenue,
# MAGIC     SUM(s.Quantity * p.Price) as gross_revenue,
# MAGIC     SUM(s.TotalPrice) - SUM(s.Quantity * p.Price) as total_discount_amount,
# MAGIC     AVG(s.Discount) as avg_discount_pct,
# MAGIC     -- Time-based metrics
# MAGIC     MIN(s.SalesDate) as first_sale_date,
# MAGIC     MAX(s.SalesDate) as last_sale_date,
# MAGIC     DATEDIFF(MAX(s.SalesDate), MIN(s.SalesDate)) + 1 as days_on_market,
# MAGIC     -- Calculate velocity
# MAGIC     SUM(s.Quantity) * 1.0 / NULLIF(DATEDIFF(MAX(s.SalesDate), MIN(s.SalesDate)) + 1, 0) as daily_sales_velocity
# MAGIC   FROM commercial.base.sales s
# MAGIC   JOIN commercial.base.products p ON s.ProductID = p.ProductID
# MAGIC   LEFT JOIN commercial.base.categories c ON p.CategoryID = c.CategoryID
# MAGIC   GROUP BY s.ProductID, p.ProductName, c.CategoryName, p.Price
# MAGIC ),
# MAGIC dataset_date_range AS (
# MAGIC   SELECT 
# MAGIC     MAX(SalesDate) as max_date,
# MAGIC     MIN(SalesDate) as min_date
# MAGIC   FROM commercial.base.sales
# MAGIC ),
# MAGIC product_growth AS (
# MAGIC   SELECT 
# MAGIC     s.ProductID,
# MAGIC     -- Compare recent vs historical performance (last 30 days of data vs prior 30 days)
# MAGIC     SUM(CASE WHEN DATEDIFF((SELECT max_date FROM dataset_date_range), s.SalesDate) <= 30 THEN s.Quantity ELSE 0 END) as recent_quantity,
# MAGIC     SUM(CASE WHEN DATEDIFF((SELECT max_date FROM dataset_date_range), s.SalesDate) BETWEEN 31 AND 60 THEN s.Quantity ELSE 0 END) as prior_quantity
# MAGIC   FROM commercial.base.sales s
# MAGIC   GROUP BY s.ProductID
# MAGIC )
# MAGIC SELECT 
# MAGIC   pp.*,
# MAGIC   COALESCE(pg.recent_quantity, 0) as recent_30d_quantity,
# MAGIC   COALESCE(pg.prior_quantity, 0) as prior_30d_quantity,
# MAGIC   CASE 
# MAGIC     WHEN pg.prior_quantity > 0 
# MAGIC     THEN ROUND((pg.recent_quantity - pg.prior_quantity) * 100.0 / pg.prior_quantity, 2)
# MAGIC     ELSE NULL
# MAGIC   END as growth_rate_pct,
# MAGIC   -- Profitability metrics
# MAGIC   ROUND(pp.total_revenue / pp.total_units_sold, 2) as revenue_per_unit,
# MAGIC   ROUND(pp.total_discount_amount / NULLIF(pp.gross_revenue, 0) * 100, 2) as discount_impact_pct,
# MAGIC   -- Portfolio classification (BCG Matrix: 2x2 matrix based on market share & growth)
# MAGIC   -- Market Share proxy: Revenue (High = top 50%, Low = bottom 50%)
# MAGIC   -- Market Growth proxy: Recent growth rate (High = positive growth, Low = negative/zero growth)
# MAGIC   CASE 
# MAGIC     -- Stars: High revenue + High growth (top 50% revenue AND positive growth)
# MAGIC     WHEN pp.total_revenue >= (SELECT PERCENTILE(total_revenue, 0.5) FROM product_performance)
# MAGIC          AND COALESCE((pg.recent_quantity - pg.prior_quantity) * 100.0 / NULLIF(pg.prior_quantity, 0), 0) > 0
# MAGIC     THEN 'Star'
# MAGIC     -- Cash Cows: High revenue + Low/negative growth (top 50% revenue AND non-positive growth)
# MAGIC     WHEN pp.total_revenue >= (SELECT PERCENTILE(total_revenue, 0.5) FROM product_performance)
# MAGIC          AND COALESCE((pg.recent_quantity - pg.prior_quantity) * 100.0 / NULLIF(pg.prior_quantity, 0), 0) <= 0
# MAGIC     THEN 'Cash Cow'
# MAGIC     -- Question Marks: Low revenue + High growth (bottom 50% revenue AND positive growth)
# MAGIC     WHEN pp.total_revenue < (SELECT PERCENTILE(total_revenue, 0.5) FROM product_performance)
# MAGIC          AND COALESCE((pg.recent_quantity - pg.prior_quantity) * 100.0 / NULLIF(pg.prior_quantity, 0), 0) > 0
# MAGIC     THEN 'Question Mark'
# MAGIC     -- Dogs: Low revenue + Low/negative growth (bottom 50% revenue AND non-positive growth)
# MAGIC     ELSE 'Dog'
# MAGIC   END as bcg_classification
# MAGIC FROM product_performance pp
# MAGIC LEFT JOIN product_growth pg ON pp.ProductID = pg.ProductID
# MAGIC ORDER BY pp.total_revenue DESC;
# MAGIC
# MAGIC SELECT * FROM product_portfolio LIMIT 50;

# COMMAND ----------

# DBTITLE 1,Product Portfolio Summary
# Portfolio Classification Summary
portfolio_summary = spark.sql("""
  SELECT 
    bcg_classification,
    COUNT(*) as product_count,
    ROUND(SUM(total_revenue), 2) as total_revenue,
    ROUND(SUM(total_revenue) * 100.0 / SUM(SUM(total_revenue)) OVER (), 2) as revenue_percentage,
    ROUND(AVG(daily_sales_velocity), 2) as avg_daily_velocity,
    ROUND(AVG(growth_rate_pct), 2) as avg_growth_rate,
    ROUND(AVG(discount_impact_pct), 2) as avg_discount_impact
  FROM product_portfolio
  GROUP BY bcg_classification
  ORDER BY total_revenue DESC
""")

display(portfolio_summary)

# Category Performance
category_performance = spark.sql("""
  SELECT 
    CategoryName,
    COUNT(*) as product_count,
    ROUND(SUM(total_revenue), 2) as total_revenue,
    ROUND(AVG(daily_sales_velocity), 2) as avg_velocity,
    ROUND(AVG(avg_discount_pct), 2) as avg_discount,
    ROUND(SUM(total_units_sold), 0) as total_units_sold
  FROM product_portfolio
  WHERE CategoryName IS NOT NULL
  GROUP BY CategoryName
  ORDER BY total_revenue DESC
""")

display(category_performance)

# COMMAND ----------

# DBTITLE 1,Objective 5 Header
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 6. Objective 5: Geographic & Sales Force Analysis
# MAGIC
# MAGIC ### 6.1 Regional Sales Performance

# COMMAND ----------

# DBTITLE 1,Geographic Analysis
# MAGIC %sql
# MAGIC -- Geographic Sales Analysis
# MAGIC CREATE OR REPLACE TEMP VIEW geographic_analysis AS
# MAGIC SELECT 
# MAGIC   c.CityID,
# MAGIC   COUNT(DISTINCT s.CustomerID) as customer_count,
# MAGIC   COUNT(DISTINCT s.SalesID) as transaction_count,
# MAGIC   SUM(s.Quantity) as total_units_sold,
# MAGIC   ROUND(SUM(s.TotalPrice), 2) as total_revenue,
# MAGIC   ROUND(AVG(s.TotalPrice), 2) as avg_transaction_value,
# MAGIC   ROUND(SUM(s.TotalPrice) / COUNT(DISTINCT s.CustomerID), 2) as revenue_per_customer,
# MAGIC   COUNT(DISTINCT s.ProductID) as unique_products_sold
# MAGIC FROM commercial.base.sales s
# MAGIC JOIN commercial.base.customers c ON s.CustomerID = c.CustomerID
# MAGIC WHERE c.CityID IS NOT NULL
# MAGIC GROUP BY c.CityID
# MAGIC ORDER BY total_revenue DESC;
# MAGIC
# MAGIC SELECT * FROM geographic_analysis LIMIT 50;

# COMMAND ----------

# DBTITLE 1,Sales Team Analysis Header
# MAGIC %md
# MAGIC ### 6.2 Sales Team Performance Analysis

# COMMAND ----------

# DBTITLE 1,Sales Team Performance
# MAGIC %sql
# MAGIC -- Sales Team Performance Analysis
# MAGIC CREATE OR REPLACE TEMP VIEW salesteam_analysis AS
# MAGIC WITH sales_performance AS (
# MAGIC   SELECT 
# MAGIC     s.SalesPersonID,
# MAGIC     e.FirstName,
# MAGIC     e.LastName,
# MAGIC     e.HireDate,
# MAGIC     DATEDIFF(DATE('2018-05-31'), e.HireDate) as tenure_days,
# MAGIC     ROUND(DATEDIFF(DATE('2018-05-31'), e.HireDate) / 365.25, 1) as tenure_years,
# MAGIC     COUNT(DISTINCT s.SalesID) as total_transactions,
# MAGIC     COUNT(DISTINCT s.CustomerID) as unique_customers,
# MAGIC     COUNT(DISTINCT s.ProductID) as unique_products_sold,
# MAGIC     SUM(s.Quantity) as total_units_sold,
# MAGIC     ROUND(SUM(s.TotalPrice), 2) as total_revenue,
# MAGIC     ROUND(AVG(s.TotalPrice), 2) as avg_transaction_value,
# MAGIC     ROUND(AVG(s.Discount), 2) as avg_discount_given,
# MAGIC     MIN(s.SalesDate) as first_sale_date,
# MAGIC     MAX(s.SalesDate) as last_sale_date
# MAGIC   FROM commercial.base.sales s
# MAGIC   JOIN commercial.base.employees e ON s.SalesPersonID = e.EmployeeID
# MAGIC   WHERE s.SalesPersonID IS NOT NULL
# MAGIC   GROUP BY s.SalesPersonID, e.FirstName, e.LastName, e.HireDate
# MAGIC )
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   ROUND(total_revenue / NULLIF(tenure_years, 0), 2) as revenue_per_year,
# MAGIC   ROUND(total_transactions / NULLIF(tenure_years, 0), 1) as transactions_per_year,
# MAGIC   ROUND(total_revenue / total_transactions, 2) as revenue_per_transaction,
# MAGIC   -- Performance tier
# MAGIC   NTILE(4) OVER (ORDER BY total_revenue DESC) as performance_quartile,
# MAGIC   CASE 
# MAGIC     WHEN NTILE(4) OVER (ORDER BY total_revenue DESC) = 1 THEN 'Top Performer'
# MAGIC     WHEN NTILE(4) OVER (ORDER BY total_revenue DESC) = 2 THEN 'Above Average'
# MAGIC     WHEN NTILE(4) OVER (ORDER BY total_revenue DESC) = 3 THEN 'Average'
# MAGIC     ELSE 'Below Average'
# MAGIC   END as performance_tier
# MAGIC FROM sales_performance
# MAGIC ORDER BY total_revenue DESC;
# MAGIC
# MAGIC SELECT * FROM salesteam_analysis LIMIT 50;

# COMMAND ----------

# DBTITLE 1,Sales Team Summary
# Sales Team Performance Summary
team_summary = spark.sql("""
  SELECT 
    performance_tier,
    COUNT(*) as salesperson_count,
    ROUND(AVG(total_revenue), 2) as avg_revenue,
    ROUND(AVG(tenure_years), 1) as avg_tenure_years,
    ROUND(AVG(unique_customers), 1) as avg_customers,
    ROUND(AVG(avg_discount_given), 2) as avg_discount_pct
  FROM salesteam_analysis
  GROUP BY performance_tier
  ORDER BY 
    CASE performance_tier
      WHEN 'Top Performer' THEN 1
      WHEN 'Above Average' THEN 2
      WHEN 'Average' THEN 3
      ELSE 4
    END
""")

display(team_summary)

# Tenure vs Performance Correlation
tenure_performance = spark.sql("""
  SELECT 
    tenure_group,
    COUNT(*) as salesperson_count,
    ROUND(AVG(total_revenue), 2) as avg_revenue,
    ROUND(AVG(total_transactions), 1) as avg_transactions,
    ROUND(AVG(unique_customers), 1) as avg_customers
  FROM (
    SELECT 
      *,
      CASE 
        WHEN tenure_years < 1 THEN 'Less than 1 year'
        WHEN tenure_years < 3 THEN '1-3 years'
        WHEN tenure_years < 5 THEN '3-5 years'
        ELSE '5+ years'
      END as tenure_group
    FROM salesteam_analysis
  )
  GROUP BY tenure_group
  ORDER BY 
    CASE tenure_group
      WHEN 'Less than 1 year' THEN 1
      WHEN '1-3 years' THEN 2
      WHEN '3-5 years' THEN 3
      ELSE 4
    END
""")

display(tenure_performance)

# COMMAND ----------

# DBTITLE 1,Summary Section
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 7. Summary: Key Insights for Dashboard
# MAGIC
# MAGIC This section creates aggregated views optimized for dashboard consumption.

# COMMAND ----------

# DBTITLE 1,Dashboard Summary Tables
# MAGIC %sql
# MAGIC -- Create summary tables for dashboard consumption
# MAGIC
# MAGIC -- 1. Inventory Classification Summary
# MAGIC CREATE OR REPLACE TEMP VIEW dashboard_inventory_summary AS
# MAGIC SELECT 
# MAGIC   abc_category,
# MAGIC   fsn_category,
# MAGIC   ved_category,
# MAGIC   COUNT(*) as product_count,
# MAGIC   ROUND(SUM(total_revenue), 2) as total_revenue,
# MAGIC   ROUND(AVG(daily_velocity), 2) as avg_daily_velocity,
# MAGIC   SUM(stockout_cancellations) as total_stockouts
# MAGIC FROM combined_inventory_classification
# MAGIC GROUP BY abc_category, fsn_category, ved_category;
# MAGIC
# MAGIC -- 2. Customer Segment Summary
# MAGIC CREATE OR REPLACE TEMP VIEW dashboard_customer_summary AS
# MAGIC SELECT 
# MAGIC   customer_segment,
# MAGIC   COUNT(*) as customer_count,
# MAGIC   ROUND(SUM(monetary_value), 2) as total_revenue,
# MAGIC   ROUND(AVG(monetary_value), 2) as avg_lifetime_value,
# MAGIC   ROUND(AVG(frequency), 1) as avg_frequency,
# MAGIC   ROUND(AVG(recency_days), 1) as avg_recency
# MAGIC FROM rfm_analysis
# MAGIC GROUP BY customer_segment;
# MAGIC
# MAGIC -- 3. Product Portfolio Summary  
# MAGIC CREATE OR REPLACE TEMP VIEW dashboard_portfolio_summary AS
# MAGIC SELECT 
# MAGIC   bcg_classification,
# MAGIC   CategoryName,
# MAGIC   COUNT(*) as product_count,
# MAGIC   ROUND(SUM(total_revenue), 2) as total_revenue,
# MAGIC   ROUND(AVG(growth_rate_pct), 2) as avg_growth_rate
# MAGIC FROM product_portfolio
# MAGIC GROUP BY bcg_classification, CategoryName;
# MAGIC
# MAGIC -- 4. Geographic Summary
# MAGIC CREATE OR REPLACE TEMP VIEW dashboard_geographic_summary AS
# MAGIC SELECT * FROM geographic_analysis;
# MAGIC
# MAGIC -- 5. Return Analysis Summary
# MAGIC CREATE OR REPLACE TEMP VIEW dashboard_return_summary AS
# MAGIC SELECT 
# MAGIC   CategoryName,
# MAGIC   ROUND(AVG(return_rate_pct), 2) as avg_return_rate,
# MAGIC   SUM(total_refunds) as total_refund_amount,
# MAGIC   SUM(quality_issues) as quality_issues,
# MAGIC   SUM(stockout_issues) as stockout_issues
# MAGIC FROM return_analysis
# MAGIC WHERE CategoryName IS NOT NULL
# MAGIC GROUP BY CategoryName;
# MAGIC
# MAGIC SELECT 'Dashboard summary tables created successfully' as status;

# COMMAND ----------

# DBTITLE 1,Export Summary Statistics
# Generate Executive Summary Statistics
print("="*80)
print("MBA PROJECT - DESCRIPTIVE ANALYTICS EXECUTIVE SUMMARY")
print("="*80)
print()

# Overall metrics
overall_stats = spark.sql("""
  SELECT 
    COUNT(DISTINCT ProductID) as total_products,
    COUNT(DISTINCT CustomerID) as total_customers,
    COUNT(DISTINCT SalesID) as total_transactions,
    ROUND(SUM(TotalPrice), 2) as total_revenue,
    ROUND(AVG(TotalPrice), 2) as avg_transaction_value
  FROM commercial.base.sales
""").collect()[0]

print(f"Total Products: {overall_stats['total_products']:,}")
print(f"Total Customers: {overall_stats['total_customers']:,}")
print(f"Total Transactions: {overall_stats['total_transactions']:,}")
print(f"Total Revenue: ${overall_stats['total_revenue']:,.2f}")
print(f"Average Transaction Value: ${overall_stats['avg_transaction_value']:,.2f}")
print()
print("="*80)
print("INVENTORY CLASSIFICATION")
print("="*80)

abc_dist = spark.sql("SELECT abc_category, COUNT(*) as count FROM abc_analysis GROUP BY abc_category ORDER BY abc_category").collect()
for row in abc_dist:
    print(f"  {row['abc_category']}-items: {row['count']:,} products")

print()
print("="*80)
print("CUSTOMER SEGMENTATION")
print("="*80)

top_segments = spark.sql("""
  SELECT customer_segment, COUNT(*) as count, ROUND(SUM(monetary_value), 2) as revenue
  FROM rfm_analysis 
  GROUP BY customer_segment 
  ORDER BY revenue DESC 
  LIMIT 5
""").collect()

for row in top_segments:
    print(f"  {row['customer_segment']}: {row['count']:,} customers, ${row['revenue']:,.2f} revenue")

print()
print("="*80)
print("ANALYSIS COMPLETE - Ready for Dashboard Creation")
print("="*80)

# COMMAND ----------

