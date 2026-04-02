# Blackjack Variant Issues and Fixes Needed

## Current Issues Identified

### 1. European Blackjack
- **Issue**: The implementation mentions "Dealer takes hole card after players finish" but doesn't properly implement this rule
- **Fix Needed**: The dealer should receive both cards at the start but not check for blackjack until after players have made all decisions

### 2. Vegas Strip
- **Issue**: The description says "dealer stands on soft 17" but the dealer logic doesn't consider soft vs hard 17
- **Fix Needed**: Implement proper soft 17 detection and respect the `dealerHitsSoft17` property

### 3. Atlantic City
- **Issue**: The description says "late surrender allowed" but the surrender logic isn't properly implemented for this variant
- **Fix Needed**: Ensure late surrender is properly enabled for Atlantic City variant

### 4. Missing Vegas Downtown Mode
- **Issue**: Mentioned in rules but not available as a game mode
- **Fix Needed**: Add Vegas Downtown as a selectable game mode with appropriate rules

### 5. Soft 17 Logic Issues
- **Issue**: The `dealerHitsSoft17` property exists but is never set to `true` and the dealer logic doesn't use it
- **Fix Needed**:
  1. Set `dealerHitsSoft17 = true` for appropriate variants
  2. Modify dealer logic to check for soft 17 when `dealerHitsSoft17 = true`

### 6. General Dealer Logic Problems
- **Issue**: Dealer only checks if total < 17, doesn't distinguish between hard and soft 17
- **Fix Needed**: Add function to detect soft hands and modify dealer decision logic accordingly

## Authentic Rules for Reference

### European Blackjack
- No hole card until after players make decisions
- Usually 6 decks
- Dealer stands on all 17s

### Vegas Strip
- Typically 4-6 decks
- Dealer stands on soft 17 (as mentioned in current implementation)
- Liberal doubling and splitting rules

### Vegas Downtown
- Single deck
- Dealer hits soft 17 (house edge increasing rule)
- Usually more favorable player rules in other aspects

### Atlantic City
- 8 decks
- Dealer stands on soft 17
- Late surrender available
- Double after split usually allowed

### Proper Soft 17 Detection
A hand is "soft 17" when it contains an Ace that can be counted as 11 without busting. For example:
- Ace-6 (can be 7 or 17)
- Ace-Ace-Ace-Ace-3 (can be 7 or 17)
- Ace-2-4 (can be 7 or 17)

The dealer should hit on soft 17 only if `dealerHitsSoft17 = true`.