# AI-Store Application API

This document outlines the functions available to programmatically control the AI-Store web application. The core of the navigation is handled by a single function, `setPage()`, which accepts a JavaScript object to determine the view and necessary parameters.

---

### Core Function: `setPage(pageObject)`

This is the primary function for all navigation and view changes within the application.

-   **`pageObject`**: An object that specifies which page to render.

---

### 1. Viewing All Products

To display the main product listing page.

-   **Function Call**: `setPage({ name: 'productList' })`
-   **Description**: Renders the `ProductList` component, showing all available items in the store. This is the default/home view.
-   **Parameters**: None.

---

### 2. Viewing a Specific Product's Details

To display the detailed view for a single product.

-   **Function Call**: `setPage({ name: 'productDetail', productId: 'prod_id' })`
-   **Description**: Renders the `ProductDetail` component for the specified product.
-   **Parameters**:
    -   `productId` (string, required): The unique identifier for the product (e.g., `'prod_001'`).

---

### 3. Viewing the Shopping Cart

To display the contents of the user's shopping cart.

-   **Function Call**: `setPage({ name: 'cart' })`
-   **Description**: Renders the `Cart` component, showing all items that have been added, the quantity of each, and the total price.
-   **Parameters**: None.

---

### 4. Adding an Item to the Cart

To add a product to the shopping cart. This action is handled by the `addToCart(product)` function, which is available within the `ProductList` and `ProductDetail` components.

-   **Function Call**: `addToCart(productObject)`
-   **Description**: Adds one unit of the specified product to the cart. If the product is already in the cart, its quantity is incremented.
-   **Parameters**:
    -   `productObject` (object, required): The full JavaScript object of the product to add. This object must be retrieved from the `productData` source first.

---

### 5. Removing an Item from the Cart

To remove a product entirely from the shopping cart. This action is handled by the `removeFromCart(productId)` function, available within the `Cart` component.

-   **Function Call**: `removeFromCart('prod_id')`
-   **Description**: Removes all quantities of a product from the cart, regardless of how many were added.
-   **Parameters**:
    -   `productId` (string, required): The unique identifier for the product to remove.