# services/redistribution.py
from utils.helpers import geo_distance
import pandas as pd


def find_recipients(recipients, product, origin):
    """
    Find top 3 recipient entities (kirana stores or NGOs) that accept the given product
    and are closest to the origin location.

    Args:
        recipients (list of dict): Each dict has keys including 'location' and 'accepted_categories'.
        product (str): Name of the product to be redistributed.
        origin (str): City name of the origin (e.g., 'Delhi').

    Returns:
        list of dict: Top 3 recipients with added 'distance_km'.
    """
    # Filter recipients that accept this product category
    eligible = []
    for rec in recipients:
        # accepted_categories is a pipe-separated string
        accepted = [cat.strip().lower() for cat in rec.get('accepted_categories', '').split('|')]
        if product.lower() in accepted:
            # Compute distance
            dist = geo_distance(origin, rec['location'])
            rec_copy = rec.copy()
            rec_copy['distance_km'] = round(dist, 2)
            eligible.append(rec_copy)

    # Sort eligible by ascending distance
    sorted_list = sorted(eligible, key=lambda r: r['distance_km'])

    # Return top 3 (or fewer if less available)
    return sorted_list[:3]


def redistribute_item(product_name: str, recipient: str, inventory_df: pd.DataFrame) -> dict:
    """
    Redistribute an item from current inventory to a recipient.
    Returns updated metrics after redistribution.
    """
    try:
        # Find the item in inventory
        item_mask = inventory_df['Product'].str.strip() == product_name.strip()
        if not item_mask.any():
            raise ValueError(f"Product '{product_name}' not found in inventory")

        item_idx = inventory_df[item_mask].index[0]
        item = inventory_df.loc[item_idx].copy()

        # Load recipients data
        try:
            recipients_df = pd.read_csv('data/recipients.csv')
        except Exception as e:
            raise ValueError(f"Failed to load recipients data: {str(e)}")

        # Verify recipient exists
        if recipient not in recipients_df['Name'].values:
            raise ValueError(f"Recipient '{recipient}' not found in records")

        # Remove item from inventory
        inventory_df.drop(item_idx, inplace=True)

        try:
            # Save updated inventory
            inventory_df.to_csv('data/mock_inventory.csv', index=False)
        except Exception as e:
            # If save fails, rollback by adding the item back
            inventory_df.loc[item_idx] = item
            raise ValueError(f"Failed to save updated inventory: {str(e)}")

        # Update recipient's received items
        recipient_row = recipients_df[recipients_df['Name'] == recipient].index[0]
        try:
            if pd.isna(recipients_df.loc[recipient_row, 'Received_Items']):
                recipients_df.loc[recipient_row, 'Received_Items'] = product_name
            else:
                # Split existing items, add new one, remove duplicates, and rejoin
                existing_items = set(recipients_df.loc[recipient_row, 'Received_Items'].split(', '))
                existing_items.add(product_name)
                recipients_df.loc[recipient_row, 'Received_Items'] = ', '.join(sorted(existing_items))

            # Save updated recipients data
            recipients_df.to_csv('data/recipients.csv', index=False)
        except Exception as e:
            # If recipient update fails, rollback inventory change
            inventory_df.loc[item_idx] = item
            inventory_df.to_csv('data/mock_inventory.csv', index=False)
            raise ValueError(f"Failed to update recipient records: {str(e)}")

        # Calculate new metrics
        total_items = len(inventory_df)
        items_at_risk = len(inventory_df[inventory_df['Days_Until_Expiry'] <= 7])
        redistribution_rate = ((total_items - items_at_risk) / total_items * 100) if total_items > 0 else 100

        return {
            'success': True,
            'metrics': {
                'total_items': total_items,
                'items_at_risk': items_at_risk,
                'redistribution_rate': f"{redistribution_rate:.1f}%"
            },
            'reload': items_at_risk == 0  # Reload if no more items at risk
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Error in redistribution: {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }
