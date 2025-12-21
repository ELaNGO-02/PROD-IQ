#!/usr/bin/env python3
"""
Complete system test - Multiple user scenarios
"""
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from llm.orchestrator import ProdIQOrchestrator


def print_result(query: str, result: dict, test_num: int):
    """Pretty print results"""
    print("\n" + "="*70)
    print(f"TEST {test_num}: {query[:60]}...")
    print("="*70)
    print(f"\n📊 TOOLS CALLED: {', '.join(result['tools_called'])}")
    print(f"\n💬 RESPONSE:\n{result['response']}\n")


def run_all_tests():
    """Run comprehensive test suite"""
    
    orchestrator = ProdIQOrchestrator()
    
    # ====================================================================
    # TEST 1: Revenue Validation (Your Original)
    # ====================================================================
    
    test_cases = [
        {
            "name": "Revenue Validation - AI SaaS",
            "query": """
                We're building an AI writing tool for content creators. Currently at $2K MRR 
                with 40 paying users at $50/month. We want to hit $10K MRR in 6 months. 
                We have $50K left and burn is $8K/month. Is this realistic? Will we survive?
            """,
            "follow_up": "What are my main competitors?"
        },
        
        # ====================================================================
        # TEST 2: Survival Analysis - Low Runway
        # ====================================================================
        {
            "name": "Survival Analysis - Critical Runway",
            "query": """
                Our fintech startup has ₹12 lakh left in the bank. Monthly burn is ₹3.5 lakh.
                We have 8 team members and are pre-revenue. We've been operating for 9 months.
                How long can we survive? Should we pivot or keep building?
            """,
            "follow_up": "What's the fastest way to generate revenue?"
        },
        
        # ====================================================================
        # TEST 3: Competitor Analysis - EdTech
        # ====================================================================
        {
            "name": "Competitor Analysis - EdTech Platform",
            "query": """
                We're building an online learning platform for coding bootcamps.
                Price: ₹15,000 per course. Currently have 120 students.
                Category: education/edtech. Who are our main competitors in India?
            """,
            "follow_up": "How do we differentiate from them?"
        },
        
        # ====================================================================
        # TEST 4: Success Probability - Pre-launch
        # ====================================================================
        {
            "name": "Success Probability - Idea Stage",
            "query": """
                We're a 3-person team planning to launch a B2B SaaS for HR analytics.
                Target price: $200/month. Target: enterprise companies with 500+ employees.
                No funding yet, bootstrapping with ₹8 lakh savings.
                What are our chances of success?
            """,
            "follow_up": "What should we focus on in the first 90 days?"
        },
        
        # ====================================================================
        # TEST 5: Break-even Timeline - Marketplace
        # ====================================================================
        {
            "name": "Break-even Analysis - Food Delivery",
            "query": """
                Our food delivery app does 800 orders/day at ₹350 average order value.
                We take 18% commission. Team of 22 people. Monthly expenses: ₹28 lakh.
                When will we break even?
            """,
            "follow_up": "Should we increase commission or reduce team size?"
        },
        
        # ====================================================================
        # TEST 6: Traction Timeline - Mobile App
        # ====================================================================
        {
            "name": "Traction Prediction - Fitness App",
            "query": """
                Launched a fitness tracking app 2 months ago. Currently 2,500 downloads,
                150 daily active users. Freemium model: ₹299/month premium.
                12 paying users so far. How long until we hit 1,000 paying users?
            """,
            "follow_up": "What's the typical conversion rate for fitness apps?"
        },
        
        # ====================================================================
        # TEST 7: Mathematical Calculations
        # ====================================================================
        {
            "name": "Math Calculation - Unit Economics",
            "query": """
                Calculate our unit economics:
                - Customer acquisition cost (CAC): ₹1,200 per customer
                - Average revenue per user (ARPU): ₹800/month
                - Average customer lifetime: 8 months
                - Monthly churn: 5%
                What's our LTV:CAC ratio?
            """,
            "follow_up": "If we reduce CAC to ₹900, how does that change things?"
        },
        
        # ====================================================================
        # TEST 8: Multi-Question Query
        # ====================================================================
        {
            "name": "Complex Multi-Part Query",
            "query": """
                We're a SaaS company with:
                - Current MRR: $15K (180 customers at $83/month average)
                - Burn: $22K/month
                - Cash: $180K
                - Team: 12 people
                - Stage: Series A fundraising
                
                Questions:
                1. Will we survive until we close the round (6 months)?
                2. Is our burn rate too high?
                3. Who are our competitors?
                4. What should our Series A ask be?
            """,
            "follow_up": "Should we reduce team size before the raise?"
        },
        
        # ====================================================================
        # TEST 9: Indian Market Specific
        # ====================================================================
        {
            "name": "Indian Market - D2C Brand",
            "query": """
                Our D2C fashion brand in India:
                - Monthly revenue: ₹18 lakh
                - Average order value: ₹2,400
                - Monthly orders: 750
                - Marketing spend: ₹6 lakh/month
                - Team: 8 people
                
                Are we growing fast enough to be venture-backable?
            """,
            "follow_up": "What metrics do Indian VCs look for in D2C?"
        },
        
        # ====================================================================
        # TEST 10: Psychology/Founder Support
        # ====================================================================
        {
            "name": "Founder Psychology - Crisis Mode",
            "query": """
                I'm a solo founder, bootstrapped, been working on this for 18 months.
                Revenue: $3K/month. Enough to pay myself $1.5K/month.
                I'm exhausted and thinking of shutting down.
                Should I keep going or get a job?
            """,
            "follow_up": "How do I know if it's time to quit?"
        },
        
        # ====================================================================
        # TEST 11: Pricing Strategy
        # ====================================================================
        {
            "name": "Pricing Optimization",
            "query": """
                We have 3 pricing tiers:
                - Basic: $29/month (400 users)
                - Pro: $79/month (80 users)
                - Enterprise: $299/month (12 users)
                
                Should we simplify to 2 tiers? Or increase prices across the board?
            """,
            "follow_up": "What's the impact of a 20% price increase?"
        },
        
        # ====================================================================
        # TEST 12: Pivot Decision
        # ====================================================================
        {
            "name": "Pivot or Persevere",
            "query": """
                Built a B2C app, got to 5K users but can't monetize.
                Pivoting to B2B SaaS targeting same problem but for enterprises.
                Current revenue: $800/month (B2C ads)
                Potential B2B price: $500/month
                
                Is this pivot worth it? We have 4 months of runway left.
            """,
            "follow_up": "How do we validate the B2B market quickly?"
        }
    ]
    
    # ====================================================================
    # RUN ALL TESTS
    # ====================================================================
    
    print("\n" + "🚀"*35)
    print("PROD-IQ COMPREHENSIVE TEST SUITE")
    print("🚀"*35 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"🧪 TEST {i}/{len(test_cases)}: {test['name']}")
        print(f"{'='*70}")
        
        # Main query
        print(f"\n📝 QUERY: {test['query'].strip()[:100]}...\n")
        result = orchestrator.process_query(test['query'])
        print_result(test['query'], result, i)
        
        # Follow-up
        if test.get('follow_up'):
            print(f"\n💬 FOLLOW-UP: {test['follow_up']}")
            result2 = orchestrator.process_query(test['follow_up'])
            print(f"\n📊 TOOLS: {', '.join(result2['tools_called'])}")
            print(f"\n💬 RESPONSE:\n{result2['response']}\n")
        
        print("\n✅ TEST COMPLETE\n")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    
    print("\n" + "="*70)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*70)
    print(f"\nTotal tests run: {len(test_cases)}")
    print("Check responses above for quality and accuracy.")


def run_single_test(test_num: int = 1):
    """Run a single test by number"""
    orchestrator = ProdIQOrchestrator()
    
    tests = {
        1: """We're building an AI writing tool for content creators. Currently at $2K MRR 
                with 40 paying users at $50/month. We want to hit $10K MRR in 6 months. 
                We have $50K left and burn is $8K/month. Is this realistic? Will we survive?""",
        2: "Fintech, ₹12L left, ₹3.5L burn, 8 people, pre-revenue, 9 months old",
        3: "EdTech coding bootcamp, ₹15K per course, 120 students",
        4: "B2B HR analytics SaaS, 3-person team, $200/month price, no funding",
        5: "Food delivery, 800 orders/day, ₹350 AOV, 18% commission, 22 people",
        6: "Fitness app, 2,500 downloads, 150 DAU, 12 paying users",
        7: "Unit economics: CAC ₹1,200, ARPU ₹800, 8 months lifetime",
        8: "SaaS $15K MRR, $22K burn, $180K cash, 12 people, Series A",
        9: "D2C fashion, ₹18L revenue, ₹2.4K AOV, 750 orders/month",
        10: "Solo founder, $3K/month, 18 months, thinking of quitting",
        11: "3 pricing tiers: $29 (400), $79 (80), $299 (12 users)",
        12: "B2C to B2B pivot, 5K users, $800/month, 4 months runway"
    }
    
    query = tests.get(test_num, tests[1])
    print(f"\n🧪 Running Test {test_num}: {query}\n")
    
    result = orchestrator.process_query(query)
    print_result(query, result, test_num)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test: python test_complete_system.py 5
        test_num = int(sys.argv[1])
        run_single_test(test_num)
    else:
        # Run all tests
        run_all_tests()
