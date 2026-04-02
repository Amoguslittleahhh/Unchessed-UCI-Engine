# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a complete **Blackjack Royale** game implemented in a single HTML file. The game features:

- Full blackjack rules with multiple game modes (Classic 3:2, Euro no hole card)
- Betting system with ALL IN functionality
- Sound effects via Web Audio API
- Game over screen with run statistics
- Clean, minimalistic green felt design with animations

## File Structure

- `blackjack.html` - Single-file blackjack game (main implementation)
- `testing file.py` - Unused Python test file (can be ignored)

## Development

Open `blackjack.html` directly in any web browser to play. No server or build tools required.

## Game Modes

- **Classic**: Dealer gets hole card, blackjack pays 3:2
- **Euro**: Dealer gets hole card after player turns, blackjack pays 2:1

## Controls

- **Bet buttons**: Select chip value to place bets
- **ALL IN**: Bet all remaining balance
- **DEAL**: Start round with current bet
- **HIT**: Take another card
- **STAND**: End turn and reveal dealer
- **DOUBLE**: Double bet and take one card
- **INSURANCE**: Side bet against dealer blackjack
