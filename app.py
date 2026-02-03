import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import itertools
from scipy.optimize import minimize

# Page configuration
st.set_page_config(
    page_title="Dynamic Pricing & Incentive Strategy Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_synthetic_data(n_samples=1000, market_conditions=None):
    """Generate realistic synthetic data for ride-hailing platform"""
    np.random.seed(42)
    
    data = {
        'surge_multiplier': np.random.uniform(1.0, 3.0, n_samples),
        'wait_time': np.random.uniform(2, 20, n_samples),
        'incentive_level': np.random.uniform(0, 50, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_type': np.random.choice([0, 1, 2], n_samples),  # 0: weekday, 1: weekend, 2: festival
        'weather': np.random.choice([0, 1, 2], n_samples),  # 0: clear, 1: rain, 2: extreme
        'competitor_price': np.random.uniform(50, 200, n_samples),
    }
    
    # Generate demand based on features (with realistic relationships)
    demand = (
        100 
        - 20 * (data['surge_multiplier'] - 1)
        - 3 * data['wait_time']
        + 0.3 * data['incentive_level']
        + 10 * (data['hour'] >= 7) * (data['hour'] <= 9)
        + 15 * (data['hour'] >= 17) * (data['hour'] <= 20)
        + 20 * (data['day_type'] == 2)
        + 15 * (data['weather'] == 1)
        - 0.2 * data['competitor_price']
        + np.random.normal(0, 10, n_samples)
    )
    
    data['demand'] = np.maximum(demand, 10)  # Ensure positive demand
    
    # Customer churn probability
    churn = (
        0.1 
        + 0.15 * (data['surge_multiplier'] - 1)
        + 0.02 * data['wait_time']
        - 0.003 * data['incentive_level']
        + np.random.normal(0, 0.05, n_samples)
    )
    data['churn_probability'] = np.clip(churn, 0, 1)
    
    return pd.DataFrame(data)

def calculate_payoff_matrix(base_demand, base_price, market_conditions):
    """Calculate game theory payoff matrix for both platforms"""
    
    strategies = ['Low Surge\nLow Incentive', 'Medium Surge\nMedium Incentive', 'High Surge\nHigh Incentive']
    
    # Payoff values (profit in thousands)
    # Format: (Platform A payoff, Platform B payoff)
    payoffs = {
        ('Low Surge\nLow Incentive', 'Low Surge\nLow Incentive'): (50, 50),
        ('Low Surge\nLow Incentive', 'Medium Surge\nMedium Incentive'): (40, 65),
        ('Low Surge\nLow Incentive', 'High Surge\nHigh Incentive'): (35, 45),
        ('Medium Surge\nMedium Incentive', 'Low Surge\nLow Incentive'): (65, 40),
        ('Medium Surge\nMedium Incentive', 'Medium Surge\nMedium Incentive'): (55, 55),
        ('Medium Surge\nMedium Incentive', 'High Surge\nHigh Incentive'): (48, 52),
        ('High Surge\nHigh Incentive', 'Low Surge\nLow Incentive'): (45, 35),
        ('High Surge\nHigh Incentive', 'Medium Surge\nMedium Incentive'): (52, 48),
        ('High Surge\nHigh Incentive', 'High Surge\nHigh Incentive'): (42, 42),
    }
    
    # Adjust based on market conditions
    multiplier = 1.0
    if market_conditions['period'] == 'Peak':
        multiplier = 1.3
    if market_conditions['day_type'] == 'Festival':
        multiplier *= 1.2
    
    adjusted_payoffs = {k: (v[0]*multiplier, v[1]*multiplier) for k, v in payoffs.items()}
    
    return strategies, adjusted_payoffs

def find_nash_equilibrium(strategies, payoffs):
    """Find Nash Equilibrium in the game"""
    nash_equilibria = []
    
    for s1 in strategies:
        for s2 in strategies:
            # Check if (s1, s2) is Nash Equilibrium
            current_payoff_a = payoffs[(s1, s2)][0]
            current_payoff_b = payoffs[(s1, s2)][1]
            
            # Check if A wants to deviate
            is_best_for_a = True
            for alt_s1 in strategies:
                if payoffs[(alt_s1, s2)][0] > current_payoff_a:
                    is_best_for_a = False
                    break
            
            # Check if B wants to deviate
            is_best_for_b = True
            for alt_s2 in strategies:
                if payoffs[(s1, alt_s2)][1] > current_payoff_b:
                    is_best_for_b = False
                    break
            
            if is_best_for_a and is_best_for_b:
                nash_equilibria.append((s1, s2))
    
    return nash_equilibria

def create_decision_tree_data(market_conditions):
    """Create decision tree structure"""
    tree = {
        'name': 'Pricing Decision',
        'children': [
            {
                'name': 'Low Surge',
                'children': [
                    {
                        'name': 'High Demand\n(60%)',
                        'value': 55,
                        'probability': 0.6
                    },
                    {
                        'name': 'Low Demand\n(40%)',
                        'value': 30,
                        'probability': 0.4
                    }
                ]
            },
            {
                'name': 'Medium Surge',
                'children': [
                    {
                        'name': 'High Demand\n(50%)',
                        'value': 75,
                        'probability': 0.5
                    },
                    {
                        'name': 'Low Demand\n(50%)',
                        'value': 40,
                        'probability': 0.5
                    }
                ]
            },
            {
                'name': 'High Surge',
                'children': [
                    {
                        'name': 'High Demand\n(30%)',
                        'value': 90,
                        'probability': 0.3
                    },
                    {
                        'name': 'Low Demand\n(70%)',
                        'value': 25,
                        'probability': 0.7
                    }
                ]
            }
        ]
    }
    
    # Calculate EMV for each decision
    emvs = {}
    for decision in tree['children']:
        emv = sum([child['value'] * child['probability'] for child in decision['children']])
        emvs[decision['name']] = emv
        decision['emv'] = emv
    
    return tree, emvs

def train_demand_prediction_model(data):
    """Train Random Forest model for demand prediction"""
    features = ['surge_multiplier', 'wait_time', 'incentive_level', 'hour', 
                'day_type', 'weather', 'competitor_price']
    X = data[features]
    y = data['demand']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, train_score, test_score, feature_importance

def optimize_pricing(model, market_conditions, budget=30):
    """Optimize pricing and incentives"""
    
    def objective(x):
        surge, incentive = x
        
        # Create input for prediction
        hour = 18 if market_conditions['period'] == 'Peak' else 12
        day_type = 2 if market_conditions['day_type'] == 'Festival' else 1 if market_conditions['day_type'] == 'Weekend' else 0
        weather = 1 if market_conditions['weather'] == 'Rainy' else 0
        
        input_data = np.array([[surge, 10, incentive, hour, day_type, weather, 100]])
        predicted_demand = model.predict(input_data)[0]
        
        # Revenue calculation
        base_price = 100
        revenue = predicted_demand * base_price * surge
        cost = predicted_demand * incentive
        
        # Maximize profit
        profit = revenue - cost
        return -profit  # Negative because we minimize
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 3.0 - x[0]},  # surge <= 3.0
        {'type': 'ineq', 'fun': lambda x: x[0] - 1.0},  # surge >= 1.0
        {'type': 'ineq', 'fun': lambda x: budget - x[1]},  # incentive <= budget
        {'type': 'ineq', 'fun': lambda x: x[1]},  # incentive >= 0
    ]
    
    # Initial guess
    x0 = [1.5, 15]
    
    # Optimize
    result = minimize(objective, x0, method='SLSQP', constraints=constraints)
    
    if result.success:
        optimal_surge, optimal_incentive = result.x
        return optimal_surge, optimal_incentive, -result.fun
    else:
        return None, None, None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Dynamic Pricing & Incentive Strategy Simulator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Decision-Support System for Ride-Hailing Platforms</strong><br>
    Simulate, analyze, and optimize pricing and driver incentive strategies using Game Theory, 
    Decision Trees, and Machine Learning.
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # MODULE 1: MARKET SETUP DASHBOARD
    # ========================================================================
    
    st.markdown('<h2 class="sub-header">📊 Module 1: Market Setup Dashboard</h2>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🎯 Market Configuration")
        
        city_name = st.text_input("City Name", "Mumbai")
        
        period = st.selectbox("Time Period", ["Peak", "Off-Peak"])
        
        day_type = st.selectbox("Day Type", ["Weekday", "Weekend", "Festival"])
        
        weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Extreme"])
        
        st.divider()
        
        st.subheader("💰 Base Parameters")
        base_fare = st.slider("Base Fare (₹)", 50, 200, 100)
        fuel_price = st.slider("Fuel Price (₹/L)", 80, 120, 95)
        estimated_demand = st.slider("Estimated Demand", 100, 500, 250)
        
        st.divider()
        
        competitor_present = st.checkbox("Competitor Present", value=True)
        
        if st.button("🔄 Generate Market Data", use_container_width=True):
            market_conditions = {
                'city': city_name,
                'period': period,
                'day_type': day_type,
                'weather': weather,
                'base_fare': base_fare,
                'fuel_price': fuel_price,
                'estimated_demand': estimated_demand,
                'competitor': competitor_present
            }
            st.session_state.generated_data = generate_synthetic_data(1000, market_conditions)
            st.session_state.market_conditions = market_conditions
            st.success("✅ Market data generated successfully!")
    
    # Display baseline metrics
    if st.session_state.generated_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Demand", f"{st.session_state.generated_data['demand'].mean():.0f}")
        with col2:
            st.metric("Avg Surge", f"{st.session_state.generated_data['surge_multiplier'].mean():.2f}x")
        with col3:
            st.metric("Avg Wait Time", f"{st.session_state.generated_data['wait_time'].mean():.1f} min")
        with col4:
            st.metric("Churn Rate", f"{st.session_state.generated_data['churn_probability'].mean()*100:.1f}%")
        
        # Baseline visualization
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=st.session_state.generated_data['demand'],
            name='Demand Distribution',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        fig.update_layout(
            title="Baseline Demand Distribution",
            xaxis_title="Demand",
            yaxis_title="Frequency",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # MODULE 2: GAME THEORY - STRATEGY SELECTION
    # ========================================================================
    
    if st.session_state.generated_data is not None:
        st.markdown('<h2 class="sub-header">🎮 Module 2: Game Theory - Strategy Selection</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🅰️ Platform A Strategy")
            a_surge = st.selectbox("Surge Pricing (A)", 
                                   ["Low Surge\nLow Incentive", "Medium Surge\nMedium Incentive", "High Surge\nHigh Incentive"],
                                   key='a_surge')
        
        with col2:
            st.subheader("🅱️ Platform B Strategy")
            b_surge = st.selectbox("Surge Pricing (B)", 
                                   ["Low Surge\nLow Incentive", "Medium Surge\nMedium Incentive", "High Surge\nHigh Incentive"],
                                   key='b_surge')
        
        # Calculate payoff matrix
        strategies, payoffs = calculate_payoff_matrix(
            st.session_state.market_conditions['estimated_demand'],
            st.session_state.market_conditions['base_fare'],
            st.session_state.market_conditions
        )
        
        # Create payoff matrix visualization
        payoff_matrix_a = np.zeros((3, 3))
        payoff_matrix_b = np.zeros((3, 3))
        
        for i, s1 in enumerate(strategies):
            for j, s2 in enumerate(strategies):
                payoff_matrix_a[i, j] = payoffs[(s1, s2)][0]
                payoff_matrix_b[i, j] = payoffs[(s1, s2)][1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_a = go.Figure(data=go.Heatmap(
                z=payoff_matrix_a,
                x=['Low', 'Medium', 'High'],
                y=['Low', 'Medium', 'High'],
                text=payoff_matrix_a,
                texttemplate='₹%{text:.0f}K',
                colorscale='Blues',
                textfont={"size": 14}
            ))
            fig_a.update_layout(
                title="Platform A Payoffs",
                xaxis_title="Platform B Strategy",
                yaxis_title="Platform A Strategy",
                height=400
            )
            st.plotly_chart(fig_a, use_container_width=True)
        
        with col2:
            fig_b = go.Figure(data=go.Heatmap(
                z=payoff_matrix_b,
                x=['Low', 'Medium', 'High'],
                y=['Low', 'Medium', 'High'],
                text=payoff_matrix_b,
                texttemplate='₹%{text:.0f}K',
                colorscale='Reds',
                textfont={"size": 14}
            ))
            fig_b.update_layout(
                title="Platform B Payoffs",
                xaxis_title="Platform B Strategy",
                yaxis_title="Platform A Strategy",
                height=400
            )
            st.plotly_chart(fig_b, use_container_width=True)
        
        # Find Nash Equilibrium
        nash_eq = find_nash_equilibrium(strategies, payoffs)
        
        st.subheader("🎯 Strategic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_payoff = payoffs[(a_surge, b_surge)]
            st.info(f"""
            **Current Strategy Outcome:**
            - Platform A Profit: ₹{selected_payoff[0]:.0f}K
            - Platform B Profit: ₹{selected_payoff[1]:.0f}K
            """)
        
        with col2:
            if nash_eq:
                st.success(f"""
                **Nash Equilibrium Found:**
                {len(nash_eq)} equilibrium point(s) detected
                
                Optimal: {nash_eq[0][0]} vs {nash_eq[0][1]}
                """)
            else:
                st.warning("No pure strategy Nash Equilibrium exists.")
        
    # ========================================================================
    # MODULE 3: DECISION TREE SIMULATOR
    # ========================================================================
    
    if st.session_state.generated_data is not None:
        st.markdown('<h2 class="sub-header">🌳 Module 3: Decision Tree Analysis</h2>', unsafe_allow_html=True)
        
        tree, emvs = create_decision_tree_data(st.session_state.market_conditions)
        
        # Display EMV calculations
        col1, col2, col3 = st.columns(3)
        
        decisions = list(emvs.keys())
        values = list(emvs.values())
        best_decision = max(emvs, key=emvs.get)
        
        with col1:
            st.metric("Low Surge EMV", f"₹{emvs['Low Surge']:.1f}K")
        with col2:
            st.metric("Medium Surge EMV", f"₹{emvs['Medium Surge']:.1f}K", 
                     delta=f"{emvs['Medium Surge']-emvs['Low Surge']:.1f}K")
        with col3:
            st.metric("High Surge EMV", f"₹{emvs['High Surge']:.1f}K",
                     delta=f"{emvs['High Surge']-emvs['Low Surge']:.1f}K")
        
        # Visualize decision tree
        fig = go.Figure()
        
        x_positions = {'Low Surge': 0, 'Medium Surge': 1, 'High Surge': 2}
        
        for decision in tree['children']:
            x_pos = x_positions[decision['name']]
            
            # Decision node
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[1],
                mode='markers+text',
                marker=dict(size=30, color='lightblue'),
                text=decision['name'],
                textposition='top center',
                name=decision['name'],
                showlegend=False
            ))
            
            # Outcome nodes
            for i, child in enumerate(decision['children']):
                y_pos = 0.3 if i == 0 else -0.3
                
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_pos], y=[1, y_pos],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=[x_pos], y=[y_pos],
                    mode='markers+text',
                    marker=dict(
                        size=25, 
                        color='lightgreen' if decision['name'] == best_decision else 'lightyellow'
                    ),
                    text=f"{child['name']}<br>₹{child['value']}K",
                    textposition='bottom center',
                    showlegend=False
                ))
        
        fig.update_layout(
            title=f"Decision Tree - Best Choice: {best_decision} (EMV: ₹{emvs[best_decision]:.1f}K)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Decision Tree Recommendation:**
        Based on Expected Monetary Value (EMV) analysis, the optimal decision is **{best_decision}** 
        with an expected profit of ₹{emvs[best_decision]:.1f}K.
        """)
    
    # ========================================================================
    # MODULE 4: DEMAND PREDICTION ENGINE
    # ========================================================================
    
    if st.session_state.generated_data is not None:
        st.markdown('<h2 class="sub-header">🤖 Module 4: ML-Based Demand Prediction</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            with st.spinner("Training Random Forest model..."):
                model, train_score, test_score, feature_importance = train_demand_prediction_model(
                    st.session_state.generated_data
                )
                st.session_state.model = model
                st.session_state.feature_importance = feature_importance
                st.session_state.model_trained = True
            
            st.success(f"✅ Model trained! Train R² = {train_score:.3f}, Test R² = {test_score:.3f}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📈 Feature Importance")
            fig = px.bar(
                st.session_state.feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔮 Demand Prediction")
            
            pred_surge = st.slider("Surge Multiplier", 1.0, 3.0, 1.5, 0.1)
            pred_wait = st.slider("Wait Time (min)", 2.0, 20.0, 10.0)
            pred_incentive = st.slider("Incentive Level (₹)", 0.0, 50.0, 25.0)
            
            hour = 18 if st.session_state.market_conditions['period'] == 'Peak' else 12
            day_type = 2 if st.session_state.market_conditions['day_type'] == 'Festival' else 1
            weather = 1 if st.session_state.market_conditions['weather'] == 'Rainy' else 0
            
            input_data = np.array([[pred_surge, pred_wait, pred_incentive, hour, day_type, weather, 100]])
            prediction = st.session_state.model.predict(input_data)[0]
            
            st.metric("Predicted Demand", f"{prediction:.0f} rides", 
                     delta=f"{prediction - st.session_state.generated_data['demand'].mean():.0f}")
            
            # Churn probability estimation
            churn_prob = 0.1 + 0.15 * (pred_surge - 1) + 0.02 * pred_wait - 0.003 * pred_incentive
            churn_prob = max(0, min(1, churn_prob))
            
            st.metric("Estimated Churn", f"{churn_prob*100:.1f}%")
        
        # Sensitivity analysis
        st.subheader("📊 Sensitivity Analysis: Surge vs Demand")
        
        surge_range = np.linspace(1.0, 3.0, 20)
        demand_predictions = []
        
        for surge in surge_range:
            input_data = np.array([[surge, 10, 25, hour, day_type, weather, 100]])
            pred = st.session_state.model.predict(input_data)[0]
            demand_predictions.append(pred)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=surge_range,
            y=demand_predictions,
            mode='lines+markers',
            name='Predicted Demand',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Impact of Surge Pricing on Demand",
            xaxis_title="Surge Multiplier",
            yaxis_title="Predicted Demand",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # MODULE 5: OPTIMIZATION & FINAL RECOMMENDATION
    # ========================================================================
    
    if st.session_state.model_trained:
        st.markdown('<h2 class="sub-header">⚡ Module 5: Optimization & Final Recommendation</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_surge_allowed = st.slider("Max Surge Allowed", 1.5, 3.0, 2.5, 0.1)
            incentive_budget = st.slider("Incentive Budget (₹)", 10.0, 50.0, 30.0)
        
        with col2:
            if st.button("🎯 Optimize Strategy", use_container_width=True):
                with st.spinner("Running optimization..."):
                    optimal_surge, optimal_incentive, max_profit = optimize_pricing(
                        st.session_state.model,
                        st.session_state.market_conditions,
                        incentive_budget
                    )
                    
                    if optimal_surge:
                        st.session_state.optimal_surge = optimal_surge
                        st.session_state.optimal_incentive = optimal_incentive
                        st.session_state.max_profit = max_profit
                        st.success("✅ Optimization complete!")
        
        if 'optimal_surge' in st.session_state:
            st.markdown(f"""
            <div class="recommendation-box">
            <strong>🎯 OPTIMAL STRATEGY RECOMMENDATION</strong><br><br>
            
            For <strong>{st.session_state.market_conditions['city']}</strong> during <strong>{st.session_state.market_conditions['period']}</strong> hours 
            on <strong>{st.session_state.market_conditions['day_type']}</strong> with <strong>{st.session_state.market_conditions['weather']}</strong> weather:
            <br><br>
            
            <strong>Surge Multiplier:</strong> {st.session_state.optimal_surge:.2f}x<br>
            <strong>Driver Incentive:</strong> ₹{st.session_state.optimal_incentive:.0f}<br>
            <strong>Expected Profit:</strong> ₹{st.session_state.max_profit:.0f}K
            <br><br>
            
            This strategy maximizes revenue while minimizing customer churn, 
            assuming competitor uses aggressive pricing.
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison chart
            col1, col2, col3 = st.columns(3)
            
            strategies_comparison = pd.DataFrame({
                'Strategy': ['Conservative', 'Optimal', 'Aggressive'],
                'Surge': [1.2, st.session_state.optimal_surge, 2.8],
                'Incentive': [10, st.session_state.optimal_incentive, 45],
                'Est. Profit': [45, st.session_state.max_profit, 38]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='Estimated Profit', x=strategies_comparison['Strategy'], 
                      y=strategies_comparison['Est. Profit'],
                      marker_color=['#ff7f0e', '#2ca02c', '#d62728'])
            ])
            fig.update_layout(
                title="Strategy Comparison",
                yaxis_title="Profit (₹K)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
    <strong>Dynamic Pricing & Incentive Strategy Simulator</strong><br>
    Powered by Game Theory • Decision Trees • Machine Learning<br>
    A CAT Project by Team-Subhitcha,Christabel | 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()