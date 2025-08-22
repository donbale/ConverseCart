# AI-Store User Guides

This document provides step-by-step instructions for common user tasks in the AI-Store.

---

### How to Find a Product

If a user wants to find a specific product or see more details about it, you should navigate to that product's detail page.

**Example User Query**: "Tell me more about the Quantum Laptop."

**Steps**:
1.  Identify the product name from the user's query ("Quantum Laptop").
2.  Look up the corresponding `productId` from the Product Catalog document. For "Quantum Laptop", the ID is `prod_001`.
3.  Use the `setPage` function to navigate to the product detail view.
4.  **Action**: Execute `setPage({ name: 'productDetail', productId: 'prod_001' })`.

---

### How to Add an Item to the Cart

When a user wants to add an item to their shopping cart, you need to call the `addToCart` function.

**Example User Query**: "Add the smart mug to my cart."

**Steps**:
1.  Identify the product the user is referring to ("smart mug").
2.  Find the full product object from the Product Catalog document. The "AI-Powered Smart Mug" has the ID `prod_004`.
3.  Retrieve the complete product object for `prod_004`.
4.  Use the `addToCart` function with the product object.
5.  **Action**: Execute `addToCart({ id: 'prod_004', name: 'AI-Powered Smart Mug', ... })`.

---

### How to See What's in the Cart

If the user asks to see their cart, you should navigate to the cart view.

**Example User Query**: "Show me my cart." or "What's in my basket?"

**Steps**:
1.  Recognize the user's intent to view their cart.
2.  Use the `setPage` function to switch to the cart view.
3.  **Action**: Execute `setPage({ name: 'cart' })`.

---

### How to Go to the Homepage

If the user wants to go back to the main product list.

**Example User Query**: "Go back to the main page." or "Show me all products."

**Steps**:
1.  Recognize the user's intent to see the full product list.
2.  Use the `setPage` function to switch to the product list view.
3.  **Action**: Execute `setPage({ name: 'productList' })`.