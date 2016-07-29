#ifndef OPQ_SYNC_MPSC_NODE_QUEUE_H_INCLUDED
#define OPQ_SYNC_MPSC_NODE_QUEUE_H_INCLUDED

#include "opaque/opq_defs.h"
#include "opaque/util/alignment.h"

#include <atomic>
#include <type_traits>

namespace opq
{
	namespace sync
	{
		struct MpscNodeHook
		{
			std::atomic<MpscNodeHook*> m_pNext;
		};

		struct MpscNodeDequeueResult
		{
			enum class Status
			{
				Ok,
				Empty,
				TryAgain
			};

			explicit MpscNodeDequeueResult( Status status, MpscNodeHook *pNode = nullptr ) :
				m_pNode( pNode ),
				m_status( status )
			{}

			explicit operator bool() const { return m_status == Status::Ok; }

			Status status() const { return m_status; }

			template<class T>
			T* get() const
			{
				static_assert( std::is_base_of<MpscNodeHook, T>::value, "T must derive from MpscNodeHook" );
				assert( ( Status::Ok == m_status ) && m_pNode );
				return static_cast<T*>( m_pNode );
			}

		private:
			MpscNodeHook *m_pNode;
			Status m_status;
		};

		struct OPQ_AVOID_FALSE_SHARING_ALIGN MpscNodeQueue : public util::Aligned<MpscNodeQueue>
		{
			enum class WasEmpty
			{
				Yes,
				No
			};

			MpscNodeQueue() :
				m_pTail( &m_node ),
				m_pHead( &m_node )				
			{
				m_node.m_pNext = nullptr;
			}

			MpscNodeQueue( const MpscNodeQueue& ) = delete;
			MpscNodeQueue& operator=( const MpscNodeQueue& ) = delete;
			MpscNodeQueue( MpscNodeQueue&& ) = delete;
			MpscNodeQueue& operator=( MpscNodeQueue&& ) = delete;

			WasEmpty enqueue( MpscNodeHook *n )
			{
				n->m_pNext.store( nullptr, std::memory_order_relaxed );
				auto pPrevTail = m_pTail.exchange( n, std::memory_order_acq_rel );
				pPrevTail->m_pNext.store( n, std::memory_order_release );

				return pPrevTail == &m_node ? WasEmpty::Yes : WasEmpty::No;
			}

			MpscNodeDequeueResult dequeue()
			{
				auto pFirstNode = m_pHead.load( std::memory_order_relaxed );
				auto pSecondNode = pFirstNode->m_pNext.load( std::memory_order_acquire );
				if ( pFirstNode == &m_node )
				{
					if ( !pSecondNode )
					{
						return MpscNodeDequeueResult( MpscNodeDequeueResult::Status::Empty );
					}

					m_pHead.store( pSecondNode, std::memory_order_relaxed );
					pFirstNode = pSecondNode;
					pSecondNode = pSecondNode->m_pNext.load( std::memory_order_acquire );
				}

				if ( pSecondNode )
				{
					m_pHead.store( pSecondNode, std::memory_order_relaxed );
					return MpscNodeDequeueResult( MpscNodeDequeueResult::Status::Ok, pFirstNode );
				}

				auto pTail = m_pTail.load( std::memory_order_relaxed );
				if ( pFirstNode != pTail )
				{
					return MpscNodeDequeueResult( MpscNodeDequeueResult::Status::TryAgain );
				}

				enqueue( &m_node );

				pSecondNode = pFirstNode->m_pNext.load( std::memory_order_relaxed );

				if ( pSecondNode )
				{
					m_pHead.store( pSecondNode, std::memory_order_relaxed );
					return MpscNodeDequeueResult( MpscNodeDequeueResult::Status::Ok, pFirstNode );
				}

				return MpscNodeDequeueResult( MpscNodeDequeueResult::Status::TryAgain );
			}

			bool empty() const
			{
				auto pFirstNode = m_pHead.load( std::memory_order_relaxed );
				auto pSecondNode = pFirstNode->m_pNext.load( std::memory_order_acquire );
				if ( pFirstNode == &m_node )
				{
					if ( !pSecondNode )
					{
						return true;
					}
				}

				return false;
			}

		private:
			OPQ_AVOID_FALSE_SHARING_ALIGN std::atomic<MpscNodeHook*> m_pTail;
			char m_padding1[OPQ_FALSE_SHARING_SIZE - sizeof( std::atomic<MpscNodeHook*> )];

			OPQ_AVOID_FALSE_SHARING_ALIGN std::atomic<MpscNodeHook*> m_pHead;
			char m_padding2[OPQ_FALSE_SHARING_SIZE - sizeof( std::atomic<MpscNodeHook*> )];

			// Node used to ensure the queue is never empty
			OPQ_AVOID_FALSE_SHARING_ALIGN MpscNodeHook m_node;
			char m_padding3[OPQ_FALSE_SHARING_SIZE - sizeof( MpscNodeHook )];
		};
	}
}

#endif
