from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import random
import datetime
import yfinance as yf

router = APIRouter()

@router.websocket("/ws/trades/{ticker}")
async def websocket_trades(websocket: WebSocket, ticker: str):
    await websocket.accept()
    ticker = ticker.upper().strip()
    
    # Establish base price using yfinance
    base_price = 150.0
    open_price = 150.0
    high_price = 152.0
    low_price = 148.0
    volume = 1000000
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            base_price = float(hist['Close'].iloc[-1])
            open_price = float(hist['Open'].iloc[-1])
            high_price = float(hist['High'].iloc[-1])
            low_price = float(hist['Low'].iloc[-1])
            volume = int(hist['Volume'].iloc[-1])
    except Exception:
        # Fallback to default mock values if yfinance fails
        pass

    current_price = base_price
    
    # Determine option strike steps based on asset value
    strike_step = 5 if current_price < 200 else (10 if current_price < 1000 else 50)

    try:
        while True:
            # Simulate slight price fluctuation (random walk)
            tick_move = random.uniform(-0.0015, 0.0015)
            current_price = round(current_price * (1 + tick_move), 2)
            
            # Update high/low bounds
            if current_price > high_price:
                high_price = current_price
            if current_price < low_price:
                low_price = current_price
                
            # Fluctuate volume
            volume += random.randint(100, 1000)
            
            # Calculate simulated percentage change
            pct_change = round(((current_price - open_price) / open_price) * 100, 2)
            
            # Generate simulated Option Chain around current price
            central_strike = round(current_price / strike_step) * strike_step
            options_chain = []
            
            for offset in range(-3, 4):
                strike = int(central_strike + offset * strike_step)
                
                # Call Option premium calculation (intrinsic + extrinsic)
                intrinsic_call = max(0.0, current_price - strike)
                extrinsic_call = max(0.10, strike_step * (1 - abs(offset) * 0.25) + random.uniform(-0.2, 0.2))
                call_ltp = round(intrinsic_call + extrinsic_call, 2)
                call_change = round(random.uniform(-5, 5), 2)
                
                # Put Option premium calculation (intrinsic + extrinsic)
                intrinsic_put = max(0.0, strike - current_price)
                extrinsic_put = max(0.10, strike_step * (1 - abs(offset) * 0.25) + random.uniform(-0.2, 0.2))
                put_ltp = round(intrinsic_put + extrinsic_put, 2)
                put_change = round(random.uniform(-5, 5), 2)
                
                options_chain.append({
                    "strike": strike,
                    "call_ltp": call_ltp,
                    "call_change": call_change,
                    "put_ltp": put_ltp,
                    "put_change": put_change
                })
                
            payload = {
                "ticker": ticker,
                "price": current_price,
                "pct_change": pct_change,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "volume": volume,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "options_chain": options_chain
            }
            
            await websocket.send_json(payload)
            await asyncio.sleep(1.5)
            
    except WebSocketDisconnect:
        # Client disconnected cleanly
        pass
    except Exception:
        # Catch-all to make sure thread cleans up
        pass
