from mcp.server.fastmcp import FastMCP

calc_mcp  = FastMCP("calculator-calc_mcp-server", host="127.0.0.1", port=8001)

@calc_mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    """
    return a + b    

@calc_mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract the second number from the first.
    """
    return a - b            

@calc_mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    """
    return a * b                

@calc_mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b        

if __name__ == "__main__":
    calc_mcp.run(transport="streamable-http")
